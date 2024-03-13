import numpy as np

def compute_rls_metrics(preds1, preds2, metrics_at=np.array([1,5,10,20,50]), rbo_p=0.9):
    """
    Compute Rank-Biased Overlap (RBO) and Jaccard similarity metrics for ranked lists.

    Args:
    - preds1 (list): List of ranked predictions (e.g., recommendations).
    - preds2 (list): List of ranked predictions for comparison.
    - metrics_at (numpy array, optional): Positions at which metrics are computed. Default is [1, 5, 10, 20, 50].
    - rbo_p (float, optional): Parameter controlling the weight decay in RBO. Default is 0.9.

    Returns:
    - dict: A dictionary containing RBO and Jaccard metrics at specified positions.
    """
    # Initialize arrays to store RBO and Jaccard scores at specified positions
    rls_rbo = np.zeros((len(metrics_at)))
    rls_jac = np.zeros((len(metrics_at)))

    for pred1,pred2 in zip(preds1,preds2):
        j = 0
        rbo_sum = 0
        for d in range(1,min(min(len(pred1),len(pred2)),max(metrics_at))+1):
            # Create sets of the first d elements from the two ranked lists
            set_pred1, set_pred2 = set(pred1[:d]), set(pred2[:d])
            
            # Calculate the intersection cardinality of the sets
            inters_card = len(set_pred1.intersection(set_pred2))

            # Update RBO sum using the formula
            rbo_sum += rbo_p**(d-1)*inters_card/d
            if d==metrics_at[j]:
                # Update RBO and Jaccard scores at the specified position
                rls_rbo[j] += (1-rbo_p)*rbo_sum/(1-rbo_p**d)
                        
                rls_jac[j] += inters_card/len(set_pred1.union(set_pred2))
                # Move to the next specified position
                j+=1
        #Check if it has stopped before cause pred1 or pred2 are shorter
        if j!=len(metrics_at):
            for k in range(j,len(metrics_at)):
                rls_rbo[k] += (1-rbo_p)*rbo_sum
                rls_jac[k] += inters_card/len(set_pred1.union(set_pred2))
     # Create dictionaries with specified positions as keys and normalized scores            
    rbo_dict = {"@"+str(k):rls_rbo[i]/len(preds1) for i,k in enumerate(metrics_at)}
    jac_dict = {"@"+str(k):rls_jac[i]/len(preds1) for i,k in enumerate(metrics_at)}
    # Return a dictionary containing RBO and Jaccard results
    return {"RLS_RBO":rbo_dict, "RLS_JAC":jac_dict}
