import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score

from ..fairness.metrics import unfairness
from ..fairness.wasserstein import WassersteinBinary

def get_metrics(parametric_sampler,
                preds_calib,
                preds_test, 
                sens_calib,
                sens_test, 
                sens_unobservable_test,
                y_test,
                mc_len=3,
                min_glob=None, 
                cu_=0.5):
    metrics_dict = {}

    fairness_trans = WassersteinBinary()
    fairness_trans.fit(preds_calib, sens_calib)

    nonparam_preds_calib = fairness_trans.transform(preds_calib, sens_calib)
    nonparam_preds_test = fairness_trans.transform(preds_test, sens_test)

    parametric_sampler.fit(X=np.array(nonparam_preds_calib), 
                           sampling_obs=len(nonparam_preds_calib)*mc_len)
    
    parametric_preds = parametric_sampler.sample(n=len(nonparam_preds_test), 
                                                 mc_samples=250)
    
    interpolator = interp1d(np.sort(nonparam_preds_test),
                            parametric_preds)
    
    param_preds_test = interpolator(nonparam_preds_test)
    if min_glob is not None:
        param_preds_test = np.maximum(min_glob, param_preds_test)

    metrics_dict['unfair'] = {}
    metrics_dict['performance'] = {}

    metrics_dict['unfair']['unfair'] = unfairness(preds_test[sens_test == 1], 
                        preds_test[sens_test != 1])
    
    metrics_dict['unfair']['unobs'] = unfairness(preds_test[sens_unobservable_test == 1], 
                        preds_test[sens_unobservable_test != 1])
    
    metrics_dict['unfair']['fair_non'] = unfairness(nonparam_preds_test[sens_test == 1], 
                          nonparam_preds_test[sens_test != 1])
    
    metrics_dict['unfair']['fair_param'] = unfairness(param_preds_test[sens_test == 1], 
                            param_preds_test[sens_test != 1])
    
    ## Set second unfairness dict

    metrics_dict['unfair']['unfair_un'] = unfairness(preds_test[sens_unobservable_test == 1], 
                        preds_test[sens_unobservable_test != 1])
    
    metrics_dict['unfair']['fair_non_un'] = unfairness(nonparam_preds_test[sens_unobservable_test == 1], 
                          nonparam_preds_test[sens_unobservable_test != 1])
    
    metrics_dict['unfair']['fair_param_un'] = unfairness(param_preds_test[sens_unobservable_test == 1], 
                            param_preds_test[sens_unobservable_test != 1])
    
    # Performance

    metrics_dict['performance']['uncorrected'] = f1_score(y_test, np.where(preds_test > cu_, 1, 0)) 
    metrics_dict['performance']['nonparam'] = f1_score(y_test, np.where(nonparam_preds_test > cu_, 1, 0))  
    metrics_dict['performance']['param'] = f1_score(y_test, np.where(param_preds_test > cu_, 1, 0)) 
        
    return metrics_dict, nonparam_preds_test, param_preds_test, parametric_sampler
    

    