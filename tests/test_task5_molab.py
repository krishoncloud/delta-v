"""Task 5 contract, leakage, staged runtime and persistence checks (offline)."""
from contextlib import chdir
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

NOTEBOOK=Path(__file__).resolve().parents[1]/'research/molab_task5.py'
SMALL={'n_modes':(4,4),'hidden_channels':8,'in_channels':16,'out_channels':4,'n_layers':1}

def notebook():
    spec=importlib.util.spec_from_file_location('task5_test',NOTEBOOK)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class Task5Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.d=notebook().app.run()[1]
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.bundle=self.d['create_example'](self.root/'study')
        self.data=self.d['validate_study'](self.bundle)
    def rehash(self):
        self.d['write_json'](self.bundle/'manifest.json',{str(p.relative_to(self.bundle)):self.d['digest'](p) for p in self.bundle.rglob('*') if p.is_file() and p.name!='manifest.json'})
    def test_exact_physics_and_reserved_reference_absence(self):
        self.assertLess(self.d['physics_check']()['max_momentum_residual'],1e-4)
        reserved=self.data['runs'].query("split=='reserved'")
        self.assertEqual(len(reserved),16)
        self.assertTrue(reserved[['target_file','kinetic_energy','pressure_rms']].isna().all().all())
    def test_bad_hash_shape_metadata_and_query_labels_rejected(self):
        path=self.bundle/'study.json';original=path.read_text()
        path.write_text(original+' ')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.d['validate_study'](self.bundle)
        path.write_text(original)
        s=json.loads(original);s['field_units']=['']*4;path.write_text(json.dumps(s));self.rehash()
        with self.assertRaisesRegex(ValueError,'units'):self.d['validate_study'](self.bundle)
        path.write_text(original)
        f=self.bundle/self.data['runs'].iloc[0].initial_file;np.save(f,np.zeros((4,4,4)));self.rehash()
        with self.assertRaisesRegex(ValueError,'Expected finite'):self.d['validate_study'](self.bundle)
    def test_setting_leakage_and_nontraining_normalization(self):
        norm=self.d['fit_normalization'](self.data)
        changed={**self.data,'runs':self.data['runs'].copy()}
        changed['runs'].loc[changed['runs'].split!='train',['amplitude','viscosity','kinetic_energy','pressure_rms']]=999
        self.assertEqual(norm,self.d['fit_normalization'](changed))
        rows=self.data['runs'].copy();a=rows[rows.split=='train'].index[0];b=rows[rows.split=='development'].index[0]
        rows.loc[b,'setting_id']=rows.loc[a,'setting_id'];rows.to_csv(self.bundle/'runs.csv',index=False);self.rehash()
        with self.assertRaisesRegex(ValueError,'Setting leakage'):self.d['validate_study'](self.bundle)
    def test_query_inputs_and_hull(self):
        norm=self.d['fit_normalization'](self.data);model=self.d['ConditionedFNO'](2,SMALL)
        q=self.data['queries'].copy();q['kinetic_energy']=0
        with self.assertRaisesRegex(ValueError,'labels'):self.d['predict_study'](model,self.data['study'],norm,q,self.bundle)
        q=self.data['queries'].copy();q.loc[0,'amplitude']=100
        with self.assertRaisesRegex(ValueError,'convex hull'):self.d['predict_study'](model,self.data['study'],norm,q,self.bundle)
    def test_preparation_stops_before_checkpoint_and_training(self):
        with chdir(self.root):
            defs=notebook().app.run(defs={'prepare_button':SimpleNamespace(value=True),'settings':SimpleNamespace(value=self.d['settings'].value)})[1]
            self.assertIn('normalization',defs);self.assertNotIn('model',defs);self.assertNotIn('training_result',defs)
    def test_full_staged_flow_and_archive_with_small_test_only_fno(self):
        # The UI has no tiny-model switch. Override only in this offline integration test.
        base=self.d
        def fake_source(config,source):
            m=base['ConditionedFNO'](2,SMALL)
            return base['tensor_state'](m.backbone),{'source':source,'epoch':0,'loaded_tensor_keys':len(base['tensor_state'](m.backbone))}
        with chdir(self.root):
            defs=notebook().app.run(defs={
                **{k:base[k] for k in ['SOURCE','SOURCE_SYSTEMS','digest','write_json','vortex_fields','physical_outcomes','physics_check','ConditionedFNO','tensor_state','count_real']},
                'prepare_button':SimpleNamespace(value=True),'load_button':SimpleNamespace(value=True),
                'train_button':SimpleNamespace(value=True),'load_source':fake_source,'FNO_CONFIG':SMALL,
                'settings':SimpleNamespace(value={'study_mode':'controlled_vortex','bundle_path':'','checkpoint_path':'','device':'cpu','steps':3,'max_seconds':60,'seed':42}),
            })[1]
        self.assertEqual(defs['training_result']['completed_steps'],3)
        self.assertTrue(defs['checks']['reload_matches']);self.assertTrue(defs['checks']['query_batch_invariant'])
        self.assertEqual(len(defs['metrics']),8);self.assertEqual(len(defs['query_predictions']),4)
        root=defs['run_dir'] if defs['run_dir'].is_absolute() else self.root/defs['run_dir']
        manifest=json.loads((root/'artifact_manifest.json').read_text())
        for name,item in manifest.items():self.assertEqual(base['digest'](root/name),item['sha256'])
        for name in ['supplied_fields.png','learning_and_outcomes.png','development_pressure.png','best_model.pt','notebook_source.py','task6_handoff.json']:
            self.assertIn(name,manifest)
        self.assertEqual((root/'notebook_source.py').read_bytes(),NOTEBOOK.read_bytes())
    def test_recovery_download_works_without_training(self):
        with chdir(self.root):
            defs=notebook().app.run(defs={
                'prepare_button':SimpleNamespace(value=True),
                'settings':SimpleNamespace(value=self.d['settings'].value),
                'recovery_button':SimpleNamespace(value=True),
            })[1]
            self.assertIn('recovery_archives',defs)
            self.assertTrue(defs['recovery_archives'][0].is_file())
            self.assertNotIn('training_result',defs)
            self.assertFalse((defs['run_dir']/'best_model.pt').exists())

    def test_source_hash_fails_closed(self):
        p=self.root/'fake.pt';p.write_bytes(b'not a checkpoint')
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            self.d['load_source']({'checkpoint_path':str(p)},self.d['SOURCE'])

if __name__=='__main__':unittest.main()
