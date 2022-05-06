# interact with system
import os
import yaml
from yaml.loader import SafeLoader
import time

# data processing
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class PaidPremiumSubscribersForecast:
    def __init__(self, params_init_yaml, params_change_yaml):
        self.init_param=self.read_yml(params_init_yaml)
        self.new_param=self.read_yml(params_change_yaml)
        
    def read_yml(self, file):
        with open(file) as f:
            data = yaml.load(f, Loader=SafeLoader)
        return data
    
    def run(self, n_months):
        self.run_free_trial_subscription(n_months)
        self.run_paid_conversion(n_months)
    
    def run_free_trial_subscription(self, n_months):
        channels = self.init_param.get('channels')
        n_channels = len(channels)
        df = np.zeros([n_months, n_channels])
        for i, c in enumerate(channels):
            init_param = self.init_param.get(c)
            x0 = init_param.get('traffic')
            cvr0 = init_param.get('cvr')
            
            new_param = self.new_param.get(c)
            new_x = new_param.get('traffic').get('new_value')
            started_new_x = new_param.get('traffic').get('started_at')
            
            new_cvr = new_param.get('cvr').get('new_value')
            started_new_cvr = new_param.get('cvr').get('started_at')

            for t in range(n_months):
                x = x0 if t < started_new_x else new_x
                cvr = cvr0 if t < started_new_cvr else new_cvr
                df[t, i] = cvr * x
                
        self.F = (df.sum(axis=1)).astype(int)
    
    def run_paid_conversion(self, n_months):
        ftp0 = self.init_param.get('ftp')
        new_ftp = self.new_param.get('ftp').get('new_value')
        started_new_ftp = self.new_param.get('ftp').get('started_at')

        retention0 = self.init_param.get('retention')
        new_retention = self.new_param.get('retention').get('new_value')
        started_new_retention = self.new_param.get('retention').get('started_at')
        
        # initiate the model run
        Y = np.zeros(n_months)
        dy = np.zeros(n_months)
        ch = np.zeros(n_months)
        
        Y[0] = 0 * self.init_param.get('y0')
        dy[0] = ftp0 * self.F[0]
        
        # run the model
        for t in range(1, n_months):
            # fit acquisition retention model
            retention = np.array(retention0 if t < started_new_retention else new_retention)
            k, b = self.fit_survival_function(retention)
            ret = np.array([self.survival_prob(t, k, b) for t in np.linspace(0, n_months, n_months)])
        
            ftp = ftp0 if t < started_new_ftp else new_ftp

            # F2P conversion this month
            dy[t] = ftp * self.F[t-1]

            # churned from previous acquisition
            ch[t] = np.sum([np.diff(ret)[i-1] * dy[t-i] for i in range(1,t+1)])

            # total paid subscribers
            Y[t] = Y[t-1] + dy[t] + ch[t]

        self.total_acquisition = Y
        self.monthly_acquisition = dy
        self.monthly_churn = ch
    

    def fit_survival_function(self, data):
        x = np.array([0, 1, 6, 12])
        y_ = np.append([1], np.array(data))
        k,b = curve_fit(self.survival_distribution, x, y_)[0]
        return k, b
    
    def survival_prob(self, x, k, b):
        p = x ** k
        c = 1 - np.exp(-b*p)
        s = 1 - c
        return s

    def survival_distribution(self, X, k, b):
        N = len(X)
        s = np.zeros(N)
        for i in range(N):
            s[i] = self.survival_prob(X[i], k, b)
        return s
    
    def plot_result(self, ax, y_data=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
        
        n_sim = len(self.total_acquisition)
        ax.plot(
            np.linspace(1, n_sim, n_sim), 
            self.total_acquisition, 
            c='tab:orange', label='Model'
        )
        
        if y_data is not None:
            n_data = len(y_data)
            ax.plot(
                np.linspace(1, n_data, n_data), 
                y_data - y_data[0], 
                '-o', c='black', alpha=0.7, label='Actual Data'
            )
            
        ax.set_xlabel('Month')
        ax.set_ylabel('#Users')
        ax.set_title('Cumulative Acquired Paid Subscribers', size=16)
        ax.legend()