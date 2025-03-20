import re
from .math_utils import is_equal, solution2answer


def compute_score(solution_str, ground_truth, continuous=False):
    """The scoring function for OpenReasoner Zero.
    
    Args:
        solution_str: the solution text
        ground_truth: the ground truth answer
        continuous: whether to use continuous scoring
    
    Returns:
        float: score between 0 and 1
    """
    try:
        # First try to find answer within <answer> tags
        pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
        matches = re.findall(pattern, solution_str)
        
        # If no matches found, try to find any \boxed{} directly
        # if not matches:
        #     pattern = re.compile(r"\\boxed{([^}]*)}")
        #     matches = re.findall(pattern, solution_str)
            
        final_answer = matches[-1] if matches else ""
        
        # If no answer found or not properly stopped, return 0
        if not final_answer:
            return 0.0
            
        # Convert both answers to standard form using solution2answer
        label = solution2answer(ground_truth)
        prefix_response = solution2answer(final_answer)
        
        # Compare using is_equal
        result = is_equal(label, prefix_response)
        
        # Return 1.0 if correct, 0.0 if wrong
        return float(result)
        
    except Exception as e:
        print(f"Error in compute_score: {e}")
        return 0.0


def val_compute_score(solution_str, ground_truth, continuous=False):
    """The scoring function for OpenReasoner Zero.
    
    Args:
        solution_str: the solution text
        ground_truth: the ground truth answer
        continuous: whether to use continuous scoring
    
    Returns:
        float: score between 0 and 1
    """
    try:
        # First try to find answer within <answer> tags
        pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
        matches = re.findall(pattern, solution_str)
        
        # If no matches found, try to find any \boxed{} directly
        if not matches:
            pattern = re.compile(r"\\boxed{([^}]*)}")
            matches = re.findall(pattern, solution_str)
            
        final_answer = matches[-1] if matches else ""
        
        # If no answer found or not properly stopped, return 0
        if not final_answer:
            return 0.0
            
        # Convert both answers to standard form using solution2answer
        label = solution2answer(ground_truth)
        prefix_response = solution2answer(final_answer)
        
        # Compare using is_equal
        result = is_equal(label, prefix_response)
        
        # Return 1.0 if correct, 0.0 if wrong
        return float(result)
        
    except Exception as e:
        print(f"Error in compute_score: {e}")
        return 0.0