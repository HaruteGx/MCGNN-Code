import itertools
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grouped_shapley import estimate_shapley


def test_additive_dummy_and_sign():
    weights = np.array([2., -3., 0., 4.])
    result = estimate_shapley(lambda m: 7+m@weights, 4, repeats=4)
    np.testing.assert_allclose(result["values"], weights, atol=1e-12)
    assert result["base"] == 7
    assert abs(result["residual"]) < 1e-12
    np.testing.assert_allclose(result["sem"], 0, atol=1e-12)


def test_pairwise_interaction_reverse_orders():
    result = estimate_shapley(lambda m: m[:,0]*m[:,1]*6+m[:,2], 3, repeats=2)
    np.testing.assert_allclose(result["values"], [3,3,1], atol=1e-12)


def test_higher_order_against_exact_subsets():
    def game(m):
        return 2*m[:,0]-m[:,1]+5*np.prod(m[:,:3],axis=1)+m[:,1]*m[:,3]
    n=4
    exact=np.zeros(n)
    for j in range(n):
        others=[i for i in range(n) if i!=j]
        for k in range(n):
            for subset in itertools.combinations(others,k):
                mask=np.zeros((2,n));mask[:,list(subset)]=1;mask[1,j]=1
                exact[j]+=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)*np.diff(game(mask))[0]
    result=estimate_shapley(game,n,repeats=2000,seed=1)
    np.testing.assert_allclose(result["values"],exact,atol=.10)
    assert abs(result["residual"])<1e-10


def test_fixed_temperature_and_background_endpoint():
    import torch
    from types import SimpleNamespace
    from grouped_shapley import CoalitionGame
    torch.manual_seed(7)
    model=SimpleNamespace(fc1=torch.nn.Linear(7,5),fc2=torch.nn.Linear(5,2))
    data={"context":torch.rand(3,1),"temperature":torch.tensor([[250.],[350.],[450.]])}
    groups=[]; projected={}
    for branch in ("environment","reactant","product"):
        layer=torch.nn.Linear(2,2)
        setattr(model,branch+"_gnn",SimpleNamespace(global_layer=layer))
        data[branch+"_local"]=torch.rand(3,2)
        data[branch+"_global"]=torch.rand(3,2)
        projected[branch]=layer(data[branch+"_global"]).detach()
        groups.append(dict(name=branch+" globals",kind="global",columns={branch:[0,1]}))
        groups.append(dict(name=branch+" local",kind="local",branch=branch))
    predictor=SimpleNamespace(model=model,data=data,projected=projected,device=torch.device("cpu"))
    game=CoalitionGame(predictor,groups,2,[0,1])
    masks=np.stack([np.zeros(6),np.ones(6),[1,0,1,0,1,0]])
    first=game(masks)
    with torch.no_grad():
        encoded=[data[b+"_local"]+torch.relu(projected[b]) for b in ("environment","reactant","product")]
        hs=model.fc2(torch.relu(model.fc1(torch.cat([*encoded,data["context"]],dim=1))))
        expected=hs[:,0]-450*hs[:,1]*.001
    np.testing.assert_allclose(first[:2],[float(expected[:2].mean()),float(expected[2])],atol=1e-6)
    data["temperature"][:2]=9999
    second=CoalitionGame(predictor,groups,2,[0,1])(masks)
    np.testing.assert_array_equal(first,second)
    result=estimate_shapley(game,6,4)
    assert abs(result["residual"])<1e-6
    np.testing.assert_allclose(game.reference_value(),first[0],atol=1e-6)


    fixed_game=CoalitionGame(predictor,groups[1:],2,[0,1],fixed_groups=groups[:1])
    rng=np.random.default_rng(1)
    reduced_masks=rng.integers(0,2,(20,5))
    full_masks=np.column_stack([np.ones(20),reduced_masks])
    np.testing.assert_allclose(fixed_game(reduced_masks),game(full_masks),atol=1e-6)
    np.testing.assert_allclose(fixed_game.reference_value(),game(np.array([[1,0,0,0,0,0]]))[0],atol=1e-6)
    np.testing.assert_allclose(fixed_game(np.ones((1,5))),game(np.ones((1,6))),atol=1e-6)
    assert abs(estimate_shapley(fixed_game,5,4)["residual"])<1e-6


def test_fingerprint_partition_preserves_original_mode():
    from grouped_shapley import attribution_partition
    groups=[dict(name="Solvent fingerprint",kind="global",columns={"environment":[0,1]}),
            dict(name="Ligand descriptor",kind="global",columns={"reactant":[2]}),
            dict(name="product local",kind="local",branch="product")]
    active,fixed=attribution_partition(groups)
    assert active==groups and fixed==[]
    active,fixed=attribution_partition(groups,"fixed")
    assert active==groups[1:] and fixed==groups[:1]


def test_fixed_context_changes_game_not_just_display():

    full=estimate_shapley(lambda m:m[:,0]*m[:,1],2,4)
    fixed=estimate_shapley(lambda m:m[:,0],1,4)
    np.testing.assert_allclose(full["values"],[.5,.5])
    np.testing.assert_allclose(fixed["values"],[1.])
