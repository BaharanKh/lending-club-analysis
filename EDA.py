import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import string


class EDA:
    def __init__(self, acc_file_name, rej_file_name):
        pd.set_option("display.max_columns", None)
        self.accepted = pd.read_csv(acc_file_name)
        self.accepted.sample(frac=0.5, replace=True, random_state=1)
        self.rejected = pd.read_csv(rej_file_name)
        display(self.accepted)
        print('accepted shape: ', self.accepted.shape)
        display(self.rejected)
        print('rejected shape: ', self.rejected.shape)
        
    def statistics(self, df):
        if df == 'acc':
            display(self.accepted.describe())
            display(self.accepted.info())
        elif df == 'rej':
            display(self.rejected.describe())
            display(self.rejected.info())
    
    def types(self, df):
        if df == 'acc':
            print(self.accepted.dtypes.value_counts())
            categorical = [feature for feature in self.accepted.columns if self.accepted[feature].dtype == "O"]
            print('categorical features: ', categorical)
        elif df == 'rej':
            print(self.rejected.dtypes.value_counts())
            categorical = [feature for feature in self.rejected.columns if self.rejected[feature].dtype == "O"]
            print('categorical features: ', categorical) 
    
    def missing_value(self, df, threshold):
        if df == 'acc':
            print(self.accepted.isnull().sum().sort_values(ascending=False))
            self.accepted = self.accepted.dropna(axis=1, thresh=len(self.accepted)*threshold)
            print(((self.accepted.isnull().sum()/len(self.accepted))*100).sort_values(ascending=False).index.values)
            self.accepted = self.accepted.drop(['id', 'url'], axis=1)
            print('Shape of the data after removing some columns: ', self.accepted.shape)
        elif df == 'rej':
            print(self.rejected.isnull().sum().sort_values(ascending=False))
            self.rejected = self.rejected.dropna( axis=1, thresh=len(self.rejected)*threshold)
            print(((self.rejected.isnull().sum()/len(self.rejected))*100).sort_values(ascending=False).index.values)
            print('Shape of the data after removing some columns: ', self.rejected.shape)
            
    def correlation(self, df):
        if df == 'acc':
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.accepted_correlations, annot=True, cmap='viridis')

            corr_pairs = self.accepted_correlations.abs().unstack().sort_values(kind="quicksort", ascending=False).dropna()
            indexes = np.array(list(corr_pairs[(corr_pairs > 0.9) & (corr_pairs < 1)].index.get_level_values(0)))
            print(corr_pairs[(corr_pairs > 0.9) & (corr_pairs < 1)])
            
        elif df == 'rej':
            self.rejected_correlations = self.rejected.corr()
            plt.figure(figsize=(8,8))
            sns.heatmap(self.rejected_correlations,cbar=True,cmap='Blues')
            
    
    
    def explore(self, df, feature):
        if df == 'acc':
            if feature == 'loan_status':
                loan_status_dummy = pd.get_dummies(self.accepted['loan_status'], drop_first=True)
                self.accepted = pd.concat([self.accepted, loan_status_dummy], axis=1)
                
                print(list(loan_status_dummy.columns))
                print(self.accepted['loan_status'].value_counts())
                plt.figure(figsize=(10,12))
                sns.countplot(x=self.accepted['loan_status'], data=self.accepted, palette='viridis')
                plt.xticks(rotation=90);

                print('Good Loans: ', len(self.accepted[(self.accepted['loan_status'] == 'Current') | (self.accepted['loan_status'] == 'Fully Paid') | (self.accepted['loan_status'] == 'In Grace Period')]) / len(self.accepted))
                
                self.accepted_correlations = self.accepted.corr()
                loan_status_correlations = self.accepted_correlations.drop(list(loan_status_dummy.columns), axis=1).loc[list(loan_status_dummy.columns)]
                
                fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(30, 30))
                for i, feature in enumerate(list(loan_status_dummy.columns)):
                    ax[int(i/2)][i%2].grid(True, linewidth=0.5, color='gray', linestyle='-')
                    ax[int(i/2)][i%2].bar(list(loan_status_correlations.loc[feature].sort_values().index), loan_status_correlations.loc[feature].sort_values(), align='center', alpha=0.5, edgecolor='black', color='purple')
                    ax[int(i/2)][i%2].set_xticklabels(list(loan_status_correlations.loc[feature].sort_values().index), rotation=90)
                    ax[int(i/2)][i%2].set_ylabel(feature)
                fig.tight_layout()
            
            elif feature == 'annual_inc':
                inc_stats = self.accepted['annual_inc'].describe()
                print(inc_stats)
                print(self.accepted.groupby('loan_status')['annual_inc'].describe().sort_values(by='count', ascending=False))
                ax = sns.boxplot(x=self.accepted['annual_inc'])
                upper_bound = inc_stats['75%'] + 1.5 * (inc_stats['75%'] - inc_stats['25%'])
                self.accepted = self.accepted[self.accepted['annual_inc']  < upper_bound]
                plt.figure(figsize=(12,5), dpi=130)
                sns.distplot(x=self.accepted['annual_inc']);
                
                sns.displot(data=self.accepted, x='annual_inc', hue='loan_status', bins=100, height=5, aspect=3, kde=True, palette='viridis');
            
            elif feature == 'loan_amnt':
                display(self.accepted['loan_amnt'].describe())
                display(self.accepted.groupby('loan_status')['loan_amnt'].describe().sort_values(by='count', ascending=False))
                plt.figure(figsize=(12,5), dpi=130)
                sns.distplot(x=self.accepted['loan_amnt']);
                sns.displot(data=self.accepted, x='loan_amnt', hue='loan_status', bins=100, height=5, aspect=3, kde=True, palette='viridis');
                ax = sns.boxplot(x=self.accepted['loan_amnt'])
                plt.figure(figsize=(12,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='loan_amnt', palette='viridis');
            
            elif feature == 'emp_length':
                print(self.accepted['emp_length'].value_counts())
                emp_length_order = [ '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
                plt.figure(figsize=(12,4))
                sns.countplot(x='emp_length',data=self.accepted,order=emp_length_order, palette='viridis');
                status_emp_length_pcnt = self.accepted.groupby(['loan_status', 'emp_length']).size().unstack(fill_value=0).stack() / self.accepted.groupby(['loan_status']).size()
                fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(35, 15))
                for i, feature in enumerate(list(self.accepted['loan_status'].value_counts().index)):
                    ax[int(i/3)][i%3].grid(True, linewidth=0.5, color='gray', linestyle='-')
                    ax[int(i/3)][i%3].bar(list(status_emp_length_pcnt[feature].index), status_emp_length_pcnt[feature], align='center', alpha=0.5, edgecolor='black', color='purple')
                    ax[int(i/3)][i%3].set_xticklabels(list(status_emp_length_pcnt[feature].index), rotation=90)
                    ax[int(i/3)][i%3].set_ylabel(feature)
                fig.tight_layout()
            
            elif feature == 'home_ownership':
                print(self.accepted['home_ownership'].value_counts())
                status_home_own_pcnt = self.accepted.groupby(['loan_status', 'home_ownership']).size().unstack(fill_value=0).stack() / self.accepted.groupby(['loan_status']).size()
                
                fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(35, 15))
                for i, feature in enumerate(list(self.accepted['loan_status'].value_counts().index)):
                    ax[int(i/3)][i%3].grid(True, linewidth=0.5, color='gray', linestyle='-')
                    ax[int(i/3)][i%3].bar(list(status_home_own_pcnt[feature].index), status_home_own_pcnt[feature], align='center', alpha=0.5, edgecolor='black', color='purple')
                    ax[int(i/3)][i%3].set_xticklabels(list(status_home_own_pcnt[feature].index), rotation=90)
                    ax[int(i/3)][i%3].set_ylabel(feature)
                fig.tight_layout()
                
            
            elif feature == 'int_rate':
                print(self.accepted['int_rate'].describe())
                print(self.accepted.groupby('loan_status')['int_rate'].describe().sort_values(by='count', ascending=False))
                plt.figure(figsize=(10,6))
                sns.boxplot(data=self.accepted, y='loan_status', x='int_rate', palette='viridis');
            
            elif feature == 'term':
                print(self.accepted['term'].value_counts())
                plt.figure(figsize=(10,12))
                x,y = 'loan_status', 'term'

                (self.accepted
                .groupby(x)[y]
                .value_counts(normalize=True)
                # .mul(100)
                .rename('percent')
                .reset_index()
                .pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
                plt.xticks(rotation=90);
            
            elif feature == 'num_actv_bc_tl':
                print(self.accepted['num_actv_bc_tl'].describe())
                print(self.accepted.groupby('loan_status')['num_actv_bc_tl'].describe().sort_values(by='count', ascending=False))
                
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['num_actv_bc_tl']);
                
                self.accepted['num_actv_bc_tl'].fillna(4, inplace=True)
                
                print(range of this variable is between 0 and 45, with mean 3.6. Most of the values are near 4. We fill the missing values with the closest number to the mean of this variable.(4))
                
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['num_actv_bc_tl']);
                
                plt.figure(figsize=(10,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='num_actv_bc_tl', palette='viridis');
                
            elif feature == 'total_acc':
                print(self.accepted['total_acc'].describe())
                print(self.accepted.groupby('loan_status')['total_acc'].describe().sort_values(by='count', ascending=False))
                plt.figure(figsize=(10,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='total_acc', palette='viridis')
                
            elif feature == 'tot_cur_bal':
                tot_cur_bal_stats = self.accepted['tot_cur_bal'].describe()
                print(tot_cur_bal_stats)    
                print(self.accepted.groupby('loan_status')['tot_cur_bal'].describe().sort_values(by='count', ascending=False))
                
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['tot_cur_bal']);
                
                self.accepted = self.accepted[self.accepted['tot_cur_bal'] < 1000000]
                print('Removed outliers with tot_cur_bal > 1000000')
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['tot_cur_bal']);
                
                sns.displot(data=self.accepted, x='tot_cur_bal', hue='loan_status', bins=100, height=5, aspect=3, kde=True, palette='viridis');
                
                plt.figure(figsize=(10,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='tot_cur_bal', palette='viridis');
            
            elif feature == 'purpose':
                print('Unique values of purpose: ', self.accepted['purpose'].unique())
                status_purpose_pcnt = self.accepted.groupby(['purpose', 'loan_status']).size().unstack(fill_value=0).stack() / self.accepted.groupby(['purpose']).size()
                fig, ax = plt.subplots(nrows=2, ncols=7, figsize=(35, 15))
                for i, feature in enumerate(list(self.accepted['purpose'].value_counts().index)):
                    ax[int(i/7)][i%7].grid(True, linewidth=0.5, color='gray', linestyle='-')
                    ax[int(i/7)][i%7].bar(list(status_purpose_pcnt[feature].index), status_purpose_pcnt[feature], align='center', alpha=0.5, edgecolor='black', color='purple')
                    ax[int(i/7)][i%7].set_xticklabels(list(status_purpose_pcnt[feature].index), rotation=90)
                    ax[int(i/7)][i%7].set_ylabel(feature)
                fig.tight_layout()
            
            elif feature == 'revol_util':
                tot_cur_bal_stats = self.accepted['revol_util'].describe()
                print(tot_cur_bal_stats)
                print(self.accepted.groupby('loan_status')['revol_util'].describe().sort_values(by='count', ascending=False))
                
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['revol_util']);
                
                self.accepted = self.accepted[accepted['revol_util'] < 150]
                
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['revol_util']);
                sns.displot(data=self.accepted, x='revol_util', hue='loan_status', bins=100, height=5, aspect=3, kde=True, palette='viridis');
                plt.figure(figsize=(10,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='revol_util', palette='viridis');
            
            elif feature == 'total_pymnt':
                print(self.accepted['total_pymnt'].describe())
                print(self.accepted.groupby('loan_status')['total_pymnt'].describe().sort_values(by='count', ascending=False))
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['total_pymnt']);
                sns.displot(data=self.accepted, x='total_pymnt', hue='loan_status', bins=100, height=5, aspect=3, kde=True, palette='viridis');
                plt.figure(figsize=(10,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='total_pymnt', palette='viridis');
            
            elif feature == 'installment':
                print(self.accepted['installment'].describe())
                print(self.accepted.groupby('loan_status')['installment'].describe().sort_values(by='count', ascending=False))
                plt.figure(figsize=(9,5), dpi=130)
                sns.distplot(x=self.accepted['installment']);
                sns.displot(data=self.accepted, x='installment', hue='loan_status', bins=100, height=5, aspect=3, kde=True, palette='viridis');
                plt.figure(figsize=(10,5), dpi=130)
                sns.boxplot(data=self.accepted, y='loan_status', x='installment', palette='viridis');
            
            elif feature == 'grade':
                print(self.accepted['grade'].value_counts())
                print()
                print(self.accepted['sub_grade'].value_counts())
                f, axes = plt.subplots(1, 2, figsize=(15,5), gridspec_kw={'width_ratios': [1, 2]})
                sns.countplot(x='grade', hue='loan_status', hue_order = ['Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off'], data=self.accepted, order=sorted(self.accepted['grade'].unique()), palette='seismic', ax=axes[0])
                sns.countplot(x='sub_grade', data=self.accepted, palette='seismic', order=sorted(self.accepted['sub_grade'].unique()), ax=axes[1], hue_order = ['Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off'])
                sns.despine()
                axes[0].set(xlabel='Grade', ylabel='Count')
                axes[0].set_title('Count of Loan Status per Grade', size=20)
                axes[1].set(xlabel='Sub Grade', ylabel='Count')
                axes[1].set_title('Count of Loan Status per Sub Grade', size=20)
                plt.tight_layout()
                
                
            elif feature == 'total_pymnt':
                print('Number of unique values for emp_title: ', self.accepted['emp_title'].nunique())
                
                
                self.accepted['emp_title'].fillna('', inplace=True)
                self.accepted['emp_title'] = self.accepted['emp_title'].apply(lambda x: string.capwords(x))
                self.accepted[self.accepted['emp_title'] == 'Rn'] = 'Registered Nurse'
                
                print('Number of unique values for emp_title after cleaning: ', self.accepted['emp_title'].nunique())
                
                print('Top 20 jobs that request loan:')
                print(self.accepted['emp_title'].value_counts()[1:21])
                      
                plt.figure(figsize=(10,5), dpi=130)
                emp_acc = self.accepted[self.accepted['emp_title'].isin(self.accepted['emp_title'].value_counts()[1:21].index.values)]
                sns.countplot(x='emp_title', hue='loan_status', hue_order = ['Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off'], data=emp_acc, palette='seismic')
                plt.xticks(rotation=90);
                
            