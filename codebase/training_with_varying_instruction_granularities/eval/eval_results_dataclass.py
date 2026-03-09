from dataclasses import dataclass, fields, asdict
from typing import List, Optional
import sqlite3
from pathlib import Path
import fcntl
import time
import pandas as pd

@dataclass
class EvalResult:
    # General
    eval_round_id: str
    train_data_setup: str
    model_type: str
    if_text_world_state_model: bool
    if_instruction_sentence_encoder: bool
    checkpoint_step: int
    task_suite_name: str
    num_chunks_for_text: int = 1
    
    # Episode Info
    # problem_id
    # see line 651 
    problem_id: int = -1 
    instruction_width: int = -1
    domain_name: str = ""
    eval_id: str = "" # dynamic
    reference_instruction_steps : int = -1
    
    
    # Evaluation Metrics
    success: bool = False
    actual_instruction_steps : int = -1
    
    
class DatabaseHelper:
    def __init__(self, db_name=Path(__file__).parent / "mini_behavior_eval_results.db"):
        self.db_name = db_name
        self.lock_file = Path(str(db_name) + ".lock")
        self.lock_fd = None
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()
        
    def check_if_exists(self, eval_result: EvalResult) -> bool:
        self._acquire_lock()
        try:
            # check all fields except success, actual_instruction_steps and eval_id
            conditions = ' AND '.join(f"{f.name} = :{f.name}" for f in fields(EvalResult) 
                                      if f.name not in ['success', 'actual_instruction_steps', 'eval_id'])
            sql = f'SELECT COUNT(*) FROM EvalResult WHERE {conditions}'
            eval_result_dict = asdict(eval_result)
            # Remove fields not in conditions
            eval_result_dict_filtered = {k: v for k, v in eval_result_dict.items() 
                                         if k not in ['success', 'actual_instruction_steps', 'eval_id']}
            self.cursor.execute(sql, eval_result_dict_filtered)
            count = self.cursor.fetchone()[0]
            return count > 0
        finally:
            self._release_lock()
        
    def _get_sql_type(self, py_type):
        if py_type == int:
            return 'INTEGER'
        elif py_type == float:
            return 'REAL'
        elif py_type == bool:
            return 'INTEGER'  # Store bool as INTEGER (0 or 1)
        else:
            return 'TEXT'
    
    def _acquire_lock(self, timeout=30):
        """Acquire file lock with timeout"""
        self.lock_fd = open(self.lock_file, 'w')
        start_time = time.time()
        while True:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError:
                if time.time() - start_time >= timeout:
                    raise TimeoutError(f"Could not acquire lock after {timeout} seconds")
                time.sleep(0.1)
    
    def _release_lock(self):
        """Release file lock"""
        if self.lock_fd:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None

    def create_table(self):
        # Generate column definitions from dataclass fields
        columns = ', '.join(f'{f.name} {self._get_sql_type(f.type)}' for f in fields(EvalResult))
        self.cursor.execute(f'CREATE TABLE IF NOT EXISTS EvalResult ({columns})')
        self.conn.commit()

    def insert_eval_result(self, eval_result: EvalResult):
        # Use named placeholders for safe insertion
        self._acquire_lock()
        try:
            headers = ','.join(f.name for f in fields(EvalResult))
            placeholders = ','.join(f':{f.name}' for f in fields(EvalResult))
            sql = f'INSERT INTO EvalResult ({headers}) VALUES ({placeholders})'
            self.cursor.execute(sql, asdict(eval_result))
            self.conn.commit()
        finally:
            self._release_lock()
        
    def delete_all_eval_results(self):
        confirmation = input("Are you sure you want to delete all eval results? Type 'yes' to confirm: ")
        if confirmation.lower() == 'yes':
            self._acquire_lock()
            try:
                self.cursor.execute('DELETE FROM EvalResult')
                self.conn.commit()
            finally:
                self._release_lock()
        else:
            print("Deletion cancelled.")

    def get_all_eval_results(self) -> list[EvalResult]:
        self._acquire_lock()
        try:
            query = 'SELECT * FROM EvalResult'
            df = pd.read_sql_query(query, self.conn)
            return df
        finally:
            self._release_lock()
            
    def get_sample_10_eval_results(self) -> list[EvalResult]:
        self._acquire_lock()
        try:
            # get success = 1 samples
            # use pd.read_sql_query for convenience
            query = 'SELECT * FROM EvalResult WHERE success = 1 ORDER BY RANDOM() LIMIT 10'
            df = pd.read_sql_query(query, self.conn)
            return df
        finally:
            self._release_lock()
            
    def remove_with_keyword(self, eval_round_id, model_type = None, use_instruction_encoding = None, text_world_state_model = None):
        if model_type is not None:
            model_type_query_part = f"model_type = '{model_type}'"
            
        if use_instruction_encoding is not None:
            instruction_encoding_query_part = f"if_instruction_sentence_encoder = {1 if use_instruction_encoding else 0}"
        if text_world_state_model is not None:
            text_world_state_model_query_part = f"if_text_world_state_model = {1 if text_world_state_model else 0}"
        query = f"DELETE FROM EvalResult WHERE eval_round_id = '{eval_round_id}'"
        if model_type is not None:
            query += f" AND {model_type_query_part}"
        if use_instruction_encoding is not None:
            query += f" AND {instruction_encoding_query_part}"
        if text_world_state_model is not None:
            query += f" AND {text_world_state_model_query_part}"
            
        # another query to show how many records will be deleted
        count_query = query.replace("DELETE FROM EvalResult", "SELECT COUNT(*) FROM EvalResult")
        self._acquire_lock()
        try:
            self.cursor.execute(count_query)
            count = self.cursor.fetchone()[0]
            print(f"Number of records to be deleted: {count}")
        finally:
            self._release_lock()
        # need to confirm 
        
        confirm_output = input(f"Are you sure to delete records with the following condition?\n{query}\n Number of records to be deleted: {count}\nType 'yes' to confirm: ")
        if confirm_output.lower() == 'yes':
            self._acquire_lock()
            try:
                self.cursor.execute(query)
                self.conn.commit()
                print(f"Deleted {count} records.")
            finally:
                self._release_lock()
        else:
            print("Deletion cancelled.")
            
    def check_if_exists(self, eval_result: EvalResult) -> bool:
        eval_dict = asdict(eval_result)
        # only take eval_round_id, train_data_setup, model_type, if_text_world_state_model, if_instruction_sentence_encoder, checkpoint_step, task_suite_name, num_chunks_for_text, problem_id, instruction_width, domain_name
        conditions = ' AND '.join(f"{k} = :{k}" for k in "eval_round_id train_data_setup model_type if_text_world_state_model if_instruction_sentence_encoder checkpoint_step task_suite_name num_chunks_for_text problem_id instruction_width domain_name".split())
        sql = f'SELECT COUNT(*) FROM EvalResult WHERE {conditions}'
        self._acquire_lock()
        try:
            self.cursor.execute(sql, {k: eval_dict[k] for k in eval_dict if k in conditions})
            count = self.cursor.fetchone()[0]
            return count > 0
        finally:
            self._release_lock()
            
    def close(self):
        if self.lock_fd:
            self._release_lock()
        self.conn.close()
        # Clean up lock file if it exists
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except:
                pass
    
    
if __name__ == "__main__":
    # Example usage
   
    db_helper = DatabaseHelper()
    results = db_helper.get_sample_10_eval_results()
    print(results)
    print(results.iloc[0])
    
    # # delete --model_type regressive --use_instruction_encoding
    # db_helper.remove_with_keyword(eval_round_id="DEC-RUSH-B", model_type="regressive", use_instruction_encoding=True)
    # # delete -model_type discrete_diffusion --use_instruction_encoding
    # db_helper.remove_with_keyword(eval_round_id="DEC-RUSH-B", model_type="discrete_diffusion", use_instruction_encoding=True)
    
    # check if already exists 
    eval_query_skeleton = dict(
        eval_round_id = 'DEC-RUSH-B',
        train_data_setup = 'single_instr_pure_H',
        model_type = 'action_head',
        if_text_world_state_model = 0,
        if_instruction_sentence_encoder = 1,
        checkpoint_step = 100000,
        task_suite_name = 'test_single_instr_id_granularity_H',
        num_chunks_for_text = 1,
        problem_id = 2481736445,
        instruction_width = 2,
        domain_name = 'sorting_books',
    )
    
    eval_result = EvalResult(**eval_query_skeleton)
    exists = db_helper.check_if_exists(eval_result)
    print(f"Eval result exists: {exists}")
    
    
    db_helper.close()