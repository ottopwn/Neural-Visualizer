import pytest
import torch

from forge import schema as S
from forge.interventions import AblateNeuron, SetBias, SetWeight, compile_interventions
from forge.introspect import IntrospectionError, MLPIntrospector
from forge.session import TrainingSettings


@pytest.fixture
def intro(session):
    session.train(TrainingSettings(epochs=5, learning_rate=0.05))
    return MLPIntrospector(session)


def neuron(intro, layer, index, ivs=(), sample=4):
    return intro.inspect(S.InspectRequest(
        ref=S.ComponentRef(kind="neuron", layer=layer, index=index),
        probe=S.Probe(sample_index=sample), interventions=list(ivs)))


def test_compile_does_not_mutate_stored_params(session):
    before = [p.weight.clone() for p in session.params]
    compile_interventions(session.spec, session.params, [SetWeight(layer=1, source=0, target=0, value=9.0)])
    assert all(torch.equal(a, p.weight) for a, p in zip(before, session.params))


def test_ablation_zeroes_output_and_shifts_downstream_exactly(intro):
    base = neuron(intro, 1, 2)
    downstream = neuron(intro, 2, 0)
    iv = AblateNeuron(layer=1, index=2)
    ablated = neuron(intro, 1, 2, [iv])
    assert ablated.ablated and ablated.value == 0.0
    assert ablated.natural_value == pytest.approx(base.value, abs=1e-6)
    after = neuron(intro, 2, 0, [iv])
    expected_shift = -downstream.weights[2] * base.value
    assert after.pre_activation - downstream.pre_activation == pytest.approx(expected_shift, abs=1e-5)


def test_set_weight_shifts_target_preactivation(intro):
    base = neuron(intro, 2, 1)
    iv = SetWeight(layer=2, source=3, target=1, value=2.5)
    after = neuron(intro, 2, 1, [iv])
    expected = (2.5 - base.weights[3]) * base.inputs[3]
    assert after.pre_activation - base.pre_activation == pytest.approx(expected, abs=1e-5)
    assert after.weight_edits[0].original == pytest.approx(base.weights[3])


def test_set_bias_and_last_edit_wins(intro):
    ivs = [SetBias(layer=3, index=0, value=1.0), SetBias(layer=3, index=0, value=-2.0)]
    out = neuron(intro, 3, 0, ivs)
    assert out.bias == pytest.approx(-2.0)
    assert out.bias_edit.value == pytest.approx(-2.0)


def test_compare_identity_and_effect(intro):
    same = intro.compare(S.ExperimentRequest(probe=S.Probe(sample_index=1)))
    assert same.delta == pytest.approx([0.0, 0.0])
    assert not same.prediction_changed and same.dataset_flip_fraction == 0.0
    assert same.boundary is not None and same.boundary.baseline == same.boundary.intervened

    # Pushing the class-1 logit bias far up must flip predictions to class 1.
    big = intro.compare(S.ExperimentRequest(probe=S.Probe(sample_index=1),
                                           interventions=[SetBias(layer=3, index=1, value=50.0)]))
    assert big.intervened.predicted_class == 1
    assert big.intervened.probabilities[1] > 0.99
    assert big.dataset_flip_fraction > 0


@pytest.mark.parametrize("iv", [
    AblateNeuron(layer=3, index=0),  # output neurons cannot be ablated
    AblateNeuron(layer=1, index=6),
    SetWeight(layer=1, source=2, target=0, value=1.0),  # input has only 2 features
    SetBias(layer=0, index=0, value=1.0),
    SetBias(layer=1, index=0, value=float("inf")),
])
def test_invalid_interventions(intro, iv):
    with pytest.raises(IntrospectionError):
        neuron(intro, 1, 0, [iv])
