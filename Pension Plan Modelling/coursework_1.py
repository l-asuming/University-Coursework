import numpy as np
import matplotlib.pyplot as plt

### Question C.1

# The second approximation uses the analytical solution for the price in between time intervals which allows for more accurate stock pricing

### Question C.2

def simulate_gbm(drift:float,volatility:float,initial_price:float,num_paths:int,total_time:float,intervals:int) -> np.matrix:
   
    if drift <= 0 or volatility <= 0 or initial_price <= 0 or total_time == 0:
        raise ValueError("Inputs have to be greater than 0")
    
    # ensures valid inputs for our model are used, catches errors python wouldn't automatically catch
    
    timestep = total_time/intervals

    delta_W_matrix = np.random.normal(0,(timestep**(1/2)),size=(num_paths,intervals))  
    price_change_matrix = np.exp((drift-(volatility**2)/2)*timestep + volatility*delta_W_matrix) 
    price_change_matrix = np.insert(price_change_matrix,0,initial_price,axis=1)

    # made a (PxN) matrix of delta W's, calculated the price change factor for each iteration of the approximation and added the initial price in the beginning of each row

    price_matrix = np.matrix.cumprod(price_change_matrix,axis=1)

    return price_matrix

    # used a cumulative product along each row to calculate the price at each interval using the iterative equation

### Question C.3

simulation_1 = simulate_gbm(0.05,0.15,1,100000,40,480)
print(simulation_1)

timesteps = np.linspace(0,40,num=481,endpoint=True)    

plt.figure(0)
plt.plot(timesteps,simulation_1[0],color="green",label="Sample paths")
plt.plot(timesteps,simulation_1[1],color="green")
plt.plot(timesteps,simulation_1[2],color="green")
plt.plot(timesteps,simulation_1[3],color="green")
plt.plot(timesteps,simulation_1[4],color="green")
plt.plot(timesteps,np.mean(simulation_1,axis=0),color="red",label="Mean path")
plt.plot(timesteps,np.percentile(simulation_1,5,axis=0),color="blue",label="5th Percentile path")
plt.plot(timesteps,np.percentile(simulation_1,95,axis=0),color="blue",label="95th Percentile path")
plt.grid()
plt.legend()
plt.xlabel(r"Timestep")
plt.ylabel(r"Price in £")
plt.title(r"Stock price simulations over time")
plt.show()
   
    # plots the first 5 simulations of the stock price, plots the mean, 5th percentile and 95th percentile paths

### Question C.4

def pension_value(price_paths:np.matrix,monthly_contribution:float) -> list:
    
    if monthly_contribution < 0:
        raise ValueError("Monthly pension contribution has to be greater than or equal to £0")
    
    stock_held_matrix = monthly_contribution/price_paths
    stock_held_matrix = np.matrix.cumsum(stock_held_matrix,axis=1)

    # creates a matrix by dividing monthly investment by the entries of the price paths matrix
    # then performs a cumulative sum to get the amount of stocks held at each timestep based on our model
    
    num_rows = price_paths.shape[0]
    num_columns = price_paths.shape[1]

    final_value_list = []

    for i in range(0,num_rows):
        final_value_list.append(stock_held_matrix[i][num_columns-1]*price_paths[i][num_columns-1])
    
    # multiplies the final amount of stock held for each price path by the final price of the corresponding price path 
    # to get the final value of the stock held and creates a list of the possible values 
    
    return final_value_list

    # note: I built this function assuming the price paths have a timestep of one month, 
    # any new function that utilises this pension_value function will also inherit this assumption

simulation_2 = simulate_gbm(0.1,0.5,10,3,1,12)
simulation_3 = simulate_gbm(0.03,0.1,100,10,2,24)
simulation_4 = simulate_gbm(0.2,1.5,100,5,0.5,6)

print(pension_value(simulation_2,1))
print(pension_value(simulation_3,100))
print(pension_value(simulation_4,5))

    # examples of different simulations and their final values

### Question C.5

simulation_1_value = pension_value(simulation_1,1000)
bins = np.arange(0,np.percentile(simulation_1_value,99)+50000,50000)

    # calculates the final pension values and makes an array of suitable bins

plt.figure(1)
plt.hist(simulation_1_value,bins=bins,edgecolor="white")
plt.xlabel(r"Final pension values (in £mn)")
plt.xticks(np.arange(0,np.percentile(simulation_1_value,99)+100000,500000))
plt.ylabel(r"Frequency")
plt.title(r"Final pension value histogram")
plt.show()

frequencies,bin_values = np.histogram(simulation_1_value,bins=bins)
max_index = np.where(frequencies == max(frequencies))[0][0]
print(f"The bin with the highest count is £{bin_values[max_index]} - £{bin_values[max_index+1]}")

    # finds the maximum frequency and its index, then uses the index of the max frequency to find the bin with the max frequency

### Question C.6

# The final pension value does not depend on the initial stock price, but rather the movement of the stock price.  
# 
# $H_n = M \sum_{i=0}^{n} \frac{1}{S_i}$ and solving the recurrence relation for $S_n$ yields:  
# 
# $S_n = S(0) \exp\left( (\mu - \frac{\sigma^{2}}{2}) n \Delta t + \sigma \sum_{k=0}^{n} \Delta W_k \right)$.  
# 
# Expressing $S_i$ in $H_n$ using our solution for the recurrence relation we get $H_n=\frac{M}{S_0} \sum_{i=0}^{n} \frac{1}{ \exp\left( (\mu - \frac{\sigma^{2}}{2}) i \Delta t + \sigma \sum_{k=0}^{i} \Delta W_k \right)}$.
# 
# As a result, in the product $V = H(T)S(T) = H_N S_N$, the $S(0)$ term cancels out and the final pension value only depends on the monthly pension contribution and the price change factors (the $\exp$ terms).

### Question C.7

def returns_probability(price_paths:np.matrix,monthly_contribution:float,target:float) -> float:

    if monthly_contribution < 0 or target < 0:
        raise ValueError("Monthly pension contribution and target value have to be greater than or equal to 0")
    
    pension_values = pension_value(price_paths,monthly_contribution) 
    target_values = [i for i in pension_values if i >= target]
    
    return len(target_values)/len(pension_values)

    # calculates the probability of making more than the target value by dividing the number of occurences
    # where the final pension value is greater than the target value by the total number of possible final investment values

print(f"Probability of making a loss is {1 - returns_probability(simulation_1,1000,481000)}")
print(f"Probability of doubling the amount invested is {returns_probability(simulation_1,1000,962000)}")
print(f"Probability of the final pension value being at least £2mn is {returns_probability(simulation_1,1000,2000000)}")

    # observed values fluctuate a little with each simulation

### Question C.8

simulation_5 = simulate_gbm(0.03,0.15,1,100000,40,480)
simulation_6 = simulate_gbm(0.07,0.15,1,100000,40,480)

    # creates more simulations with the same conditions as the simulation in C.3 except drift changing to 0.03 and 0.07

m_values1 = np.linspace(0,4000,num=201,endpoint=True)    

probabilities_1 = [returns_probability(simulation_1,i,1000000) for i in m_values1]
probabilities_5 = [returns_probability(simulation_5,i,1000000) for i in m_values1]
probabilities_6 = [returns_probability(simulation_6,i,1000000) for i in m_values1]

    # calculates the probability for 201 monthly contribution values for each simulation starting at £0 and increasing in £20 increments

plt.figure(2)
plt.plot(m_values1,probabilities_5,color="blue",label=r"$\mu$ = 0.03")
plt.plot(m_values1,probabilities_1,color="red",label=r"$\mu$ = 0.05")
plt.plot(m_values1,probabilities_6,color="green",label=r"$\mu$ = 0.07")
plt.grid()
plt.legend()
plt.xlabel(r"Monthly contribution")
plt.ylabel(r"Probability")
plt.title(r"Probability of a having a final investment value greater than £1mn against monthly pension contribution (for 40 years)")
plt.show()
   
    # plots the probabilities of having a final investment value greater than £1mn for paths with drift = 0.03, 0.05, 0.07

# $\mu$ = 0.03 crosses the 95% threshold for a monthly contribution = £3300-3320, $\mu$ = 0.05 crosses the 95% threshold for a monthly contribution = £2240-2260, $\mu$ = 0.07 crosses the 95% threshold for a monthly contribution = £1440-1460

### Question C.9

simulation_7 = simulate_gbm(0.03,0.15,1,100000,20,240)
simulation_8 = simulate_gbm(0.05,0.15,1,100000,20,240)
simulation_9 = simulate_gbm(0.07,0.15,1,100000,20,240)

    # creates more simulations with the same conditions as the simulations for C.8 except time period changing to 20 years

m_values2 = np.linspace(0,8000,num=401,endpoint=True)    

    # chose an interval of [0,8000]

probabilities_7 = [returns_probability(simulation_7,i,1000000) for i in m_values2]
probabilities_8 = [returns_probability(simulation_8,i,1000000) for i in m_values2]
probabilities_9 = [returns_probability(simulation_9,i,1000000) for i in m_values2]

    # calculates the probability for 401 monthly contribution values for each simulation starting at 0 and increasing in £20 increments

plt.figure(3)
plt.plot(m_values2,probabilities_7,color="blue",label=r"$\mu$ = 0.03")
plt.plot(m_values2,probabilities_8,color="red",label=r"$\mu$ = 0.05")
plt.plot(m_values2,probabilities_9,color="green",label=r"$\mu$ = 0.07")
plt.grid()
plt.legend()
plt.xlabel(r"Monthly contribution")
plt.ylabel(r"Probability")
plt.title(r"Probability of a having a final investment value greater than £1mn against Monthly pension contribution (for 20 years)")
plt.show()

  # plots the probabilities of having a final investment value greater than £1mn for paths with drift = 0.03, 0.05, 0.07  

# $\mu$ = 0.03 crosses the 95% threshold for a monthly contribution = £6220-6240 (£2900-£2940 more per month than compared to the 40 years case), $\mu$ = 0.05 crosses the 95% threshold for a monthly contribution = £5140-5160 (£2880-2920 more per month compared to the 40 years case), $\mu$ = 0.07 crosses the 95% threshold for a monthly contribution = £4220-4240 (£2760-2800 more per month compared to the 40 years case)
