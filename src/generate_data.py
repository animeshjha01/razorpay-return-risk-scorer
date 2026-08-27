import argparse
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import os

def generate_data(n=5000, seed=42, test_size=0.2):
    np.random.seed(seed)
    
    # Generate base fields
    order_ids = [f"order_{uuid.uuid4().hex[:10]}" for _ in range(n)]
    payment_ids = [f"pay_{uuid.uuid4().hex[:10]}" for _ in range(n)]
    
    # Dates over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    created_at = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(n)]
    
    # Amount INR (skewed log-normal)
    amount_inr = np.random.lognormal(mean=6.5, sigma=1.0, size=n).round(2)
    amount_inr = np.clip(amount_inr, 50, 50000)
    
    # Method
    methods = ['upi', 'card', 'netbanking', 'wallet', 'cod']
    method = np.random.choice(methods, p=[0.4, 0.2, 0.1, 0.05, 0.25], size=n)
    
    # Category
    categories = ['electronics', 'apparel', 'grocery', 'digital', 'home']
    category = np.random.choice(categories, p=[0.2, 0.3, 0.2, 0.15, 0.15], size=n)
    
    # Customer history
    is_new_customer = np.random.choice([0, 1], p=[0.7, 0.3], size=n)
    past_orders = np.where(is_new_customer, 0, np.random.poisson(lam=5, size=n))
    
    # past_return_rate - forced 0 for new customers
    past_return_rate = np.where(is_new_customer, 0.0, np.random.beta(a=1, b=5, size=n))
    
    # Time features
    order_hour = np.array([dt.hour for dt in created_at])
    is_weekend = np.array([1 if dt.weekday() >= 5 else 0 for dt in created_at])
    is_late_night = np.array([1 if (h < 5 or h >= 23) else 0 for h in order_hour])
    
    # Delivery and checkout
    delivery_distance_km = np.random.exponential(scale=15, size=n).round(1)
    checkout_time_sec = np.random.lognormal(mean=4.0, sigma=0.8, size=n).round(1)
    
    # Assemble df
    df = pd.DataFrame({
        'order_id': order_ids,
        'payment_id': payment_ids,
        'created_at': created_at,
        'amount_inr': amount_inr,
        'method': method,
        'category': category,
        'is_new_customer': is_new_customer,
        'past_orders': past_orders,
        'past_return_rate': past_return_rate,
        'order_hour': order_hour,
        'is_weekend': is_weekend,
        'is_late_night': is_late_night,
        'delivery_distance_km': delivery_distance_km,
        'checkout_time_sec': checkout_time_sec
    })
    
    # Risk computation
    # Weights for risk score
    z = -4.2  # Base intercept
    
    # COD is riskier
    z += np.where(df['method'] == 'cod', 1.5, 0.0)
    
    # Apparel is riskier
    z += np.where(df['category'] == 'apparel', 0.5, 0.0)
    z -= np.where(df['category'] == 'digital', 1.0, 0.0)
    
    # High past return rate is very risky
    z += df['past_return_rate'] * 5.0
    
    # New customers are slightly riskier but don't have past_return_rate.
    # The lack of past_return_rate means they get 0 from the above. 
    # To make new customers slightly risky, add a term:
    z += df['is_new_customer'] * 1.0
    
    # High amount is slightly riskier
    z += np.log1p(df['amount_inr']) * 0.1
    
    # Distance
    z += (df['delivery_distance_km'] / 50.0) * 0.5
    
    # Late night
    z += df['is_late_night'] * 0.3
    
    # Noise
    z += np.random.normal(0, 1.0, size=n)
    
    # Probabilities
    prob = 1 / (1 + np.exp(-z))
    
    # Labels
    df['returned_or_chargeback'] = np.random.binomial(1, prob)
    
    # Split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['returned_or_chargeback'])
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print(f"Generated {len(train_df)} training and {len(test_df)} test records.")
    print(f"Base return rate: {df['returned_or_chargeback'].mean():.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    
    generate_data(n=args.n, seed=args.seed, test_size=args.test_size)
