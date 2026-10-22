import json
import os
import numpy as np
import pytest
from hilserl import ExperimentConfig
from hilserl_profiles import image_shape, pixel_schema, classifier_schema, UPSTREAM_SCHEMA, UPSTREAM_CLASSIFIER
from hilserl_remote import rows_to_arrays, arrays_to_rows
from test_hilserl_upstream import config, spec, row


def test_resolution_contracts_and_transport():
    assert pixel_schema(config()) == UPSTREAM_SCHEMA
    assert classifier_schema(config()) == UPSTREAM_CLASSIFIER
    for size in (64,128,256):
        c=config(image_size=size)
        assert image_shape(c)==(9,size,size)
        assert f'3x{size}' in pixel_schema(c) and f'3x{size}' in classifier_schema(c)
        sample=row();sample.update(images=np.zeros(image_shape(c),np.uint8),next_images=np.ones(image_shape(c),np.uint8))
        restored=arrays_to_rows(rows_to_arrays([sample],True),spec(c))
        assert restored[0]['images'].shape==(9,size,size)
    with pytest.raises(ValueError): ExperimentConfig(image_size=96)


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_UPSTREAM') != '1', reason='numerical model test')
@pytest.mark.parametrize('size',[64,256])
def test_actor_learner_and_classifier_at_selected_resolution(tmp_path,size):
    from hilserl_upstream import UpstreamSAC, UpstreamClassifier
    from hilserl_data import write_npz
    c=config(image_size=size,pixel_batch_size=2,observation_cameras=['head'])
    learner=UpstreamSAC(c)
    sample=row();sample.update(images=np.zeros(image_shape(c),np.uint8),next_images=np.ones(image_shape(c),np.uint8))
    result=learner.update([sample,sample])
    assert np.isfinite(result['critic_loss'])
    assert learner.policy.action(np.zeros(22),sample['images']).shape==(6,)
    classifier=UpstreamClassifier(['head'],'cpu',image_size=size)
    images=np.stack([sample['images'],sample['next_images']])
    assert np.isfinite(classifier.train_batch(images,[0,1]))
    expected=classifier.logits(images)
    path=tmp_path/'classifier.npz'
    write_npz(path,__schema__=np.array(classifier_schema(c)),__cameras__=np.array(json.dumps(['head'])),**classifier.export())
    loaded=UpstreamClassifier.load(path,'cpu')
    np.testing.assert_allclose(loaded.logits(images),expected,rtol=1e-6)
