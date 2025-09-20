"""
Created on Mon May  1 16:06:56 2023

@author: leila Jafari Khouzani
"""

import os
import csv
import operator
import pandas as pd
import numpy as np
# =============================================================================
class ZeroEdgesCreator():
    def __init__(self):
        self.dataset_path = os.path.join(os.getcwd(), 'dataset')
        DPI_path = os.path.join(self.dataset_path, 'DPI.csv')
        DDS_path = os.path.join(self.dataset_path, 'BiDrugDrugSimilarity.tsv')
        PPI_path = os.path.join(self.dataset_path,'BiProteinProteinIdentity.tsv')       
         
        DPI = pd.read_csv(DPI_path, sep=',') 
        self.DDS = pd.read_csv(DDS_path, sep='\t')        
        self.PPI = pd.read_csv(PPI_path, sep='\t')   
        self.DPI_drugs= DPI['DrugID']
        self.DPI_proteins= DPI['UniprotID']
        self.drug_names = DPI['DrugID'].unique()
        self.protein_names = DPI['UniprotID'].unique()
        
# ============================================================================= 
    def create_zero_samples(self, out_file='Negative_scores.tsv'):
        s_ij= dict()
        i=0
        for  drug_i in self.drug_names:
            # find indexes where the drug is drug_i in DPI
            index_protein_interact = np.where(self.DPI_drugs== drug_i)   
            # with indexes find proteins interact with drug_i
            protein_interact = self.DPI_proteins[index_protein_interact[0]] 
            # remove interactd proteins in order to find non_interacted proteins
            protein_Ninteract = np.setdiff1d(self.DPI_proteins,protein_interact)
            
            # 
            col1_Ninteract = self.PPI[self.PPI["Protein1"].isin(protein_Ninteract)]
            col2_interact = col1_Ninteract[col1_Ninteract["Protein2"].isin(protein_interact)]  
            DDiS = col2_interact.drop(columns = ['Protein2'])
            identity = DDiS.groupby('Protein1')['Identity'].sum() 
            new_index =drug_i + "_" + identity.index
            identity.index = new_index
            DiP = identity.to_dict()
            s_ij.update(DiP)
            i+=1
            print(i, ": ", drug_i)
            
        s_ji =dict()
        i=0     
        for  protein_j in self.protein_names:
            index_drug_interact = np.where(self.DPI_proteins== protein_j)     
            drug_interact = self.DPI_drugs[index_drug_interact[0]] 
            drug_Ninteract = np.setdiff1d(self.DPI_drugs,drug_interact)
            
            col1_Ninteract = self.DDS[self.DDS["Drug1"].isin(drug_Ninteract)]
            col2_interact = col1_Ninteract[col1_Ninteract["Drug2"].isin(drug_interact)]  
            PPjI = col2_interact.drop(columns = ['Drug2'])
            similarity = PPjI.groupby('Drug1')['Similarity'].sum() 
            new_index = similarity.index + "_" + protein_j
            similarity.index = new_index
            PjD = similarity.to_dict()
            s_ji.update(PjD) 
            i+=1
            print(i, ": ",protein_j)
           
        sum_sij_sji = {k: s_ij.get(k, 0) + s_ji.get(k, 0) for k in set(s_ij) | set(s_ji)}   
      
        exp_S = sum_sij_sji.copy()
        exp_S.update((key, np.exp(-1*value)) for key, value in exp_S.items())
        
        Negative_scores = sorted(exp_S.items(),key=operator.itemgetter(1),reverse=True)
        out_file = os.path.join(self.dataset_path, out_file)     
        with open(out_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(Negative_scores)  
    # =============================================================================
    def evaluate_zero_samples(self, out_file):
        s_ij= dict()
        s_ji =dict()
        i=0
        for  drug_i, protein_j in zip (self.DPI_drugs, self.DPI_proteins):
            # find indexes where the drug is drug_i in DPI
            index_protein_interact = np.where(self.DPI_drugs== drug_i)   
            # by indexes find proteins interact with drug_i
            protein_interact = list(self.DPI_proteins[index_protein_interact[0]])       
            protein_interact.remove(protein_j)
            # 
            col1_protein_j = self.PPI[self.PPI["Protein1"]==protein_j]
            col2_interact = col1_protein_j[col1_protein_j["Protein2"].isin(protein_interact)]  
            PPjI = col2_interact['Identity']
            identity = PPjI.sum() 
            new_index =drug_i + "_" + protein_j
            s_ij[new_index] =  identity
            
            index_drug_interact = np.where(self.DPI_proteins== protein_j)     
            drug_interact = list(self.DPI_drugs[index_drug_interact[0]])  
            drug_interact.remove(drug_i)   
            
            col1_drug_i = self.DDS[self.DDS["Drug1"]==drug_i]
            col2_interact = col1_drug_i[col1_drug_i["Drug2"].isin(drug_interact)]  
            DDiS = col2_interact['Similarity']
            similarity = DDiS.sum()  
            s_ji[new_index] = similarity
            i+=1
            print(i, ": ",drug_i,protein_j)
        
        sum_sij_sji = {k: s_ij.get(k, 0) + s_ji.get(k, 0) for k in set(s_ij) | set(s_ji)}
            
        exp_S = sum_sij_sji.copy()
        exp_S.update((key, np.exp(-1*value)) for key, value in exp_S.items())
        Positive_edge_scores = sorted(exp_S.items(),key=operator.itemgetter(1),reverse=True)    
        
        out_file = os.path.join(self.dataset_path, out_file)
        with open(out_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(Positive_edge_scores)  
# =============================================================================
def select_unique_ids_until_limit(df, directory, thr=0.5, target_count=3225):
    df.columns = ['DrugID', 'UniprotID', 'ZeroScore']
    df_filtered = df[df['ZeroScore'] > thr]
    collected = pd.DataFrame(columns=df.columns)
    remaining = df_filtered.copy()
   
    # === First iteration: keep all unique DrugIDs and UniprotIDs ===
    unique_drugs = remaining.drop_duplicates(subset=['DrugID'], keep='first')
    unique_targets = remaining.drop_duplicates(subset=['UniprotID'], keep='first')
    first_batch = pd.concat([unique_drugs, unique_targets]).drop_duplicates()

    collected = pd.concat([collected, first_batch]).drop_duplicates()

    # Remove collected rows from remaining
    remaining = remaining.merge(collected, how='left', indicator=True)
    remaining = remaining[remaining['_merge'] == 'left_only'].drop(columns=['_merge'])
    repeat = 0
    # === Repeat to reach target_count (if needed) ===
    while len(collected) < target_count and not remaining.empty:
        repeat += 1
        unique_drugs = remaining.drop_duplicates(subset=['DrugID'], keep='first')
        unique_targets = remaining.drop_duplicates(subset=['UniprotID'], keep='first')
        new_unique = pd.concat([unique_drugs, unique_targets]).drop_duplicates()

        # Avoid duplicates
        new_unique = new_unique.merge(collected, how='left', indicator=True)
        new_unique = new_unique[new_unique['_merge'] == 'left_only'].drop(columns=['_merge'])

        collected = pd.concat([collected, new_unique]).drop_duplicates()

        # Update remaining
        remaining = remaining.merge(new_unique, how='left', indicator=True)
        remaining = remaining[remaining['_merge'] == 'left_only'].drop(columns=['_merge'])

    collected['ActionType'] = 'ZeroAction'
    collected['ActionLabel'] = 0   
    # === Save results ===
    path_neg = os.path.join(directory, "zero_"+ str(repeat+thr) +".csv")
    collected.to_csv(path_neg, index=False)

    path_remain = os.path.join(directory, "remained_"+ str(thr) +".csv")
    remaining.to_csv(path_remain, index=False)

    print(f"Final unique pairs: {len(collected)} (target: {target_count}). Remaining rows: {len(remaining)}.")

# =============================================================================
def clean_tsv_file(directory, 
                   input_file="zero_scores.tsv", 
                   output_file="edited_Zero_scores.tsv"):
    """
    Reads a TSV file, removes empty rows, replaces underscores with commas,
    and writes the cleaned lines to a new file.
    """
    input_path = os.path.join(directory, input_file)
    output_path = os.path.join(directory,  output_file)
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line:  # skip empty lines
                cleaned_line = line.replace('_', ',')
                outfile.write(cleaned_line + '\n')
    
    df = pd.read_csv(output_path, sep=',', header=None)   
    return df     

# =============================================================================
def calculate_portion_above_thresholds(df, score_column='ZeroScore', output_csv='threshold_portion_summary.csv'):
    """
    Calculates the portion of rows with score >= threshold for thresholds from 0.1 to 0.9,
    and saves the result to a CSV file.

    Parameters:
        df (pd.DataFrame): Input DataFrame with at least one column for scores.
        score_column (str): Name of the column containing the scores.
        output_csv (str): Path to the output CSV file.
    """
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []

    for thr in thresholds:
        portion = (df[score_column] >= thr).mean()
        results.append({'Threshold': thr, 'PortionAboveThreshold': portion})

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
# =============================================================================
def edit_file(directory):
    """remove all ids that are not exist in ids dataframe from df.
    This is done because some drug ids has not features and should
    remove their interactions"""

    input_path = os.path.join(directory, "Zero_scores.tsv")
    df = pd.read_csv(input_path, sep=',')
    
    drug_id_file = os.path.join(directory, 'DrugDiscriptors.csv')
    drug_ids = pd.read_csv(drug_id_file, sep=',')
    drugs_to_remain = list(drug_ids['DrugID'])
    # Drop rows where 'DrugID' values are not in 'drugs_to_drop'
    df_filtered = df[df['DrugID'].isin(drugs_to_remain)]    
    target_id_file = os.path.join(directory, 'all_uniprot_ids.csv')
    target_ids = pd.read_csv(target_id_file, sep=',')
    targets_to_remain = list(target_ids['UniprotID'])
    # Drop rows where 'UniprotID' values are not in 'targets_to_drop'
    df_filtered2 = df_filtered[df_filtered['UniprotID'].isin(targets_to_remain)]    
    
    df_filtered2.to_csv(directory+'\\cleaned.txt', index=False)
    return df_filtered
# =============================================================================
#                           main:   
# =============================================================================   
if __name__ == '__main__':
    obj = ZeroEdgesCreator()
    out_file = 'Zero_scores.tsv'
    obj.create_zero_samples(out_file)

    directory = "dataset"
    df_filtered = edit_file(directory)
    directory = os.path.join ("dataset", "zero pairs with thr")
    threshold = 0.5
    select_unique_ids_until_limit(df_filtered, directory, threshold)

    out_file = 'Positive_edge_scores.tsv'
    obj.evaluate_zero_samples(out_file)

    out_file = os.path.join(obj.dataset_path, out_file)
    df = pd.read_csv(out_file, sep=',')
    df.columns[0] = "DrugID"
    df.columns[1] = "UniprotID"
    df.columns[2] = "ZeroScore"
    calculate_portion_above_thresholds(df)
