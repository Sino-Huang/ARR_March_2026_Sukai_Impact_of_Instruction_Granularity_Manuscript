import os 
from glob import glob
from tqdm.auto import tqdm
def classify_widths_method_by_quantile(widths):
    """
    Classify widths into Low, Middle, High granularity using quantile-based binning.
    
    Args:
        widths: List of available widths for a task (e.g., [0, 1, 2, 3])
    
    Returns:
        Dictionary with keys 'Low', 'Middle', 'High' containing the classified widths
    """
    # Sort the widths to ensure proper ranking
    sorted_widths = sorted(widths)
    n = len(sorted_widths)
    
    # Calculate split indices
    q1 = n // 3  # End of first group
    q2 = 2 * n // 3  # End of second group
    
    # Assign to classes
    low = sorted_widths[:q1]
    middle = sorted_widths[q1:q2]
    high = sorted_widths[q2:]
    
    # if low is empty, assign the middle lowest element to low
    if len(low) == 0 and len(middle) > 0:
        low.append(middle[0])
    elif len(low) == 0 and len(high) > 0:
        low.append(high[0])
        
    if len(middle) == 0 and len(high) > 0:
        middle.append(high[0])
    
    return {
        'Low': low,
        'Middle': middle, 
        'High': high
    }

# Test function to verify all your tasks
def test_all_tasks():
    """Test the classification on all 20 tasks"""
    data_dir = '/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/02_intermediate/gps_data'
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    domain_name_list = [
        "laying_wood_floors",
        "preparing_salad",
        "cleaning_up_the_kitchen_only",
        "organizing_file_cabinet",
        "thawing_frozen_food",
        "making_tea",
        "opening_packages",
        "boxing_books_up_for_storage",
        "collect_misplaced_items",
        "putting_away_dishes_after_cleaning", # need to deal with compact mode for this one
        "washing_pots_and_pans",
        "cleaning_shoes",
        "installing_a_printer",
        "setting_up_candles",
        "watering_houseplants",
        "cleaning_a_car",
        "storing_food",
        "throwing_away_leftovers",
        "moving_boxes_to_storage",
        "sorting_books",
    ]
    map_id = "*"
    tasks = {}
    for domain_name in domain_name_list:
        print(f"--- Domain: {domain_name} ---")
        for width in tqdm([0,1,2,3,4,5,6,7]):
            pattern_str = os.path.join(data_dir, f"{domain_name}/gps_data_map_{map_id}_width_{width}_env_MiniGrid-*_goal_True.pkl")
            matched_files = glob(pattern_str, recursive=True)
            if len(matched_files) > 0:
                if domain_name not in tasks:
                    tasks[domain_name] = []
                tasks[domain_name].append(width)
                   
    print("Testing all tasks with (Quantile-Based Binning):")
    print("=" * 80)
    
    for task_id, widths in tasks.items():
        result = classify_widths_method_by_quantile(widths)
        print(f"Task {task_id}: {widths} -> L:{result['Low']} M:{result['Middle']} H:{result['High']}")
    
    return tasks, [classify_widths_method_by_quantile(widths) for widths in tasks.values()]

if __name__ == "__main__":
    test_all_tasks()
    
    
# Testing all tasks with (Quantile-Based Binning):
# ================================================================================
# Task laying_wood_floors: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task preparing_salad: [0, 1, 2, 3, 4] -> L:[0] M:[1, 2] H:[3, 4]
# Task cleaning_up_the_kitchen_only: [0, 1, 2, 3, 4, 5] -> L:[0, 1] M:[2, 3] H:[4, 5]
# Task organizing_file_cabinet: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task thawing_frozen_food: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task making_tea: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task opening_packages: [0, 1] -> L:[] M:[0] H:[1]
# Task boxing_books_up_for_storage: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task collect_misplaced_items: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task putting_away_dishes_after_cleaning: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task washing_pots_and_pans: [0, 1, 2, 3, 4] -> L:[0] M:[1, 2] H:[3, 4]
# Task cleaning_shoes: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task installing_a_printer: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task setting_up_candles: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task watering_houseplants: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task cleaning_a_car: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task storing_food: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task throwing_away_leftovers: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task moving_boxes_to_storage: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task sorting_books: [0, 1, 2] -> L:[0] M:[1] H:[2]