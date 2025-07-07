import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import astroquery.jplhorizons as ast
from typing import Union

### Question B.1

#Consider the interval $[E_-,E_+]=[-1,2\pi+1]$, $f(0)=-M<0$ and $f(2\pi)=2\pi-M>0$ as $M\in[0,2\pi)$
#
#As $f(E)=E-esin(E)-M$ is continuous and $-M<0<2\pi-M,\:\exists E_0\in[-1,2\pi+1]:f(E_0)=0$ by IVT

### Question B.2

#Let $E_{n+1}=esin(E_n)+M$
#
#Set $g(E)=esin(E)+M$, resulting in $g'(E)=ecos(E)$
#
#$|ecos(E)|\le e<1:\forall E\in\mathbb{R}$ as $e\in [0,1)$ for an ellipse, meaning $g(E)$ is a contraction
#
#$g([-1,2\pi+1])=[M-e,M+e]=[-1,2\pi+1)$, meaning $g(E)$ is endomorphic on this interval
# 
#Therefore by the contraction mapping theorem, $E_{n+1}=esin(E_n)+M$ converges to $E$ as $n\to\infty$    

### Question B.3

E,i,omega,Omega = sp.symbols("E,i,ω,Ω")

def eccentricity_anomaly(M:Union[float,sp.Expr],eccentricity:float,verbose:bool=False)->float:
    if M < 0 or M >= 2*sp.pi or eccentricity < 0 or eccentricity >= 1:
        raise ValueError("M is in [0,2*pi) and e is in [0,1)")

    residual = E-eccentricity*sp.sin(E)-M
    residual_p = sp.diff(residual,E)
    residual_p_p = sp.diff(residual_p,E)
    
    #defines the residual function and finds the first and second derivative of the residual function with respect to E

    E0 = 1
    E1 = float(E0 - 2*(residual.subs({E:E0})*residual_p.subs({E:E0})).evalf()/(2*(residual_p.subs({E:E0})**2)-residual.subs({E:E0})*residual_p_p.subs({E:E0})).evalf())
    
    iteration = 1

    #set our initial guess to 1 and calculates the first iteration using Halley's Method

    if verbose:
        while abs(float(residual.subs({E:E0}).evalf()))>10**-6 and iteration<=100:

            residual_value = float(residual.subs({E:E0}).evalf())
            residual_p_value = float(residual_p.subs({E:E0}).evalf())
            residual_p_p_value = float(residual_p_p.subs({E:E0}).evalf())

            print(f"Iteration:{iteration} Residual:{residual_value}")
            
            (E0,E1) = (E1,E0 - 2*(residual_value*residual_p_value)/(2*(residual_p_value**2)-residual_value*residual_p_p_value))
            
            iteration += 1
        
        if iteration>100:
            raise RuntimeError("Halley's method did not converge within 100 iterations")
                
        return E0
        
    else:
        while abs(float(residual.subs({E:E0}).evalf()))>10**-6 and iteration<=100 :
            
            residual_value = float(residual.subs({E:E0}).evalf())
            residual_p_value = float(residual_p.subs({E:E0}).evalf())
            residual_p_p_value = float(residual_p_p.subs({E:E0}).evalf())
            
            (E0,E1) = (E1,E0 - 2*(residual_value*residual_p_value)/(2*(residual_p_value**2)-residual_value*residual_p_p_value))

            iteration += 1
        
        if iteration>100:
            raise RuntimeError("Halley's method did not converge within 100 iterations")
        
        return E0
    
    #calculates multiple iterations using Halley's method with the residual tending to zero as more iterations occur
    #while loop will terminate when the residual is close to zero (or if convergence takes too long to occur)
    #the function returns the value of E that brings the residual below the threshold
    #verbose prints the iteration number and the residual for each value of E_n in our iterations

### Question B.4

print(f"Eccenticity anomaly for M=0.1pi and eccentricity=1/2 is {eccentricity_anomaly(0.1*sp.pi,1/2,verbose=True)}")
print(f"Eccenticity anomaly for M=0.3pi and eccentricity=1/2 is {eccentricity_anomaly(0.3*sp.pi,1/2,verbose=True)}")
print(f"Eccenticity anomaly for M=0.7pi and eccentricity=1/2 is {eccentricity_anomaly(0.7*sp.pi,1/2,verbose=True)}") #calculates an approximate value of E in each given case
print(f"Eccenticity anomaly for M=pi and eccentricity=1/2 is {eccentricity_anomaly(sp.pi,1/2,verbose=True)}")        #and prints out residuals for each iteration

### Question B.5

days_over_time_period_1 = np.linspace(0,1,num=400,endpoint=False)
M_values_1_pi = 2*days_over_time_period_1*np.pi

eccentricity_anomalies_1 = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,0,verbose=False) for n in days_over_time_period_1]
eccentricity_anomalies_2 = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,1/10,verbose=False) for n in days_over_time_period_1]
eccentricity_anomalies_3 = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,1/2,verbose=False) for n in days_over_time_period_1]
eccentricity_anomalies_4 = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,9/10,verbose=False) for n in days_over_time_period_1]   #creates a list of eccentricity anomalies
eccentricity_anomalies_5 = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,99/100,verbose=False) for n in days_over_time_period_1] #for multiple values of M given an eccentricity 

plt.figure(1)
plt.plot(M_values_1_pi,eccentricity_anomalies_1,color="blue",label=r"e = 0")
plt.plot(M_values_1_pi,eccentricity_anomalies_2,color="red",label=r"e = 0.1")
plt.plot(M_values_1_pi,eccentricity_anomalies_3,color="green",label=r"e = 0.5")
plt.plot(M_values_1_pi,eccentricity_anomalies_4,color="yellow",label=r"e = 0.9")
plt.plot(M_values_1_pi,eccentricity_anomalies_5,color="pink",label=r"e = 0.99")
plt.xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],[r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
plt.yticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],[r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
plt.grid()
plt.legend()
plt.xlabel(r"Mean Anomaly")
plt.ylabel(r"Eccentricity Anomaly")
plt.title(r"Eccentricity Anomaly against Mean Anomaly")
plt.show()  #plots eccentricity anomaly against mean anomaly for different eccentricities

### Question B.6

rotation_1 = sp.Matrix([[sp.cos(omega),-sp.sin(omega),0],[sp.sin(omega),sp.cos(omega),0],[0,0,1]])
rotation_2 = sp.Matrix([[1,0,0],[0,sp.cos(i),-sp.sin(i)],[0,sp.sin(i),sp.cos(i)]])
rotation_3 = sp.Matrix([[sp.cos(Omega),-sp.sin(Omega),0],[sp.sin(Omega),sp.cos(Omega),0],[0,0,1]])

rotation_composition = rotation_3*rotation_2*rotation_1 #is the transformation that takes coordinates from the orbital plane  
print(rotation_composition) #and transforms them into coordinates on the reference plane 

### Question B.7

semi_major_axis_1 = 1.523679
eccentricity_1 = 0.0934

mars_orbital_coords = sp.Matrix([semi_major_axis_1*(sp.cos(E)-eccentricity_1), semi_major_axis_1*sp.sqrt(1-eccentricity_1**2)*sp.sin(E), 0])

mars_rotation = rotation_composition.subs({omega:5.0006, i:0.0323, Omega:0.8656}) #used the orbital elements to transform coordinates from Mars' orbital plane
mars_reference_coords = mars_rotation*mars_orbital_coords                         #to coordinates on our reference plane that is centred around the sun

days_over_time_period_2 = list(np.linspace(0, 1, num=687, endpoint=False)) #used increments of 1/T to work out (t-tau)/T for each day in our interval,
days_over_time_period_2.extend(days_over_time_period_2+[0])                #3 times of periapsis occur at the very beginning, middle and very end of our interval

mars_eccentricity_anomalies = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,eccentricity_1,verbose=False) for n in days_over_time_period_2]

mars_approx_x_coords = [float(mars_reference_coords.col(0)[0].subs({E:n}).evalf()) for n in mars_eccentricity_anomalies] #calculates the approximate daily
mars_approx_y_coords = [float(mars_reference_coords.col(0)[1].subs({E:n}).evalf()) for n in mars_eccentricity_anomalies] #coordinates of Mars at 11:08 between
mars_approx_z_coords = [float(mars_reference_coords.col(0)[2].subs({E:n}).evalf()) for n in mars_eccentricity_anomalies] #2020-08-03 and 2024-05-08 using the orbital elements

mars = ast.Horizons(id="499",location="500@10", epochs={"start": "2020-08-03 11:08:00","stop": "2024-05-08 11:08:00","step": "1d"},id_type="id")

mars_true_x_coords = [float(row['x']) for row in mars.vectors()]
mars_true_y_coords = [float(row['y']) for row in mars.vectors()]
mars_true_z_coords = [float(row['z']) for row in mars.vectors()]

earth_1 = ast.Horizons(id="399",location="500@10", epochs={"start": "2020-08-03 11:08:00","stop": "2024-05-08 11:08:00","step": "1d"},id_type="id")

earth_true_x_coords = [float(row['x']) for row in earth_1.vectors()]
earth_true_y_coords = [float(row['y']) for row in earth_1.vectors()] #extracts the true daily coordinates of Mars and Earth at 11:08
earth_true_z_coords = [float(row['z']) for row in earth_1.vectors()] #between 2020-08-03 and 2024-05-08 using the JPL Horizons database 

fig = plt.figure(2, figsize=(8,6.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel(r"X (AU)")
ax.set_ylabel(r"Y (AU)")
ax.set_zlabel(r"Z (AU)")
ax.plot(mars_true_x_coords, mars_true_y_coords, mars_true_z_coords, color="red", label=r"Mars' true path")
ax.plot(mars_approx_x_coords, mars_approx_y_coords, mars_approx_z_coords, linewidth=2.5, linestyle="--", color="black", label=r"Mars' approximate path")
ax.plot(earth_true_x_coords, earth_true_y_coords, earth_true_z_coords, color="green", label=r"Earth's path")
ax.scatter(0, 0, 0, color='gold', s=50, label='Sun') 
ax.text(0, 0, 0, 'Sun', color='gold', fontsize=12)
ax.set_title(r"Paths of Mars and Earth between 2020-08-03 and 2024-05-08")
plt.legend()
plt.show() #plots the approximated postion of Mars and the true positions of Mars and Earth with reference to the Sun (that is at the origin) over our time interval

### Question B.8

distances_1 = []
for n in range(len(mars_approx_x_coords)):
    distances_1.append(np.sqrt((mars_true_x_coords[n]-mars_approx_x_coords[n])**2+(mars_true_y_coords[n]-mars_approx_y_coords[n])**2+(mars_true_z_coords[n]-mars_approx_z_coords[n])**2))
max_distance = max(distances_1)             #creates a list of distances between the approximated postion of Mars and its true position each day in our time interval,
max_index = distances_1.index(max_distance) #finds the max distance and the index of this max distance in our list of distances to help find the date this max distance occurs

print(f"Max distance between Mars' true path and its approximated path is {max_distance} AU")
print(f"Day of max distance between Mars' true path and its approximated path is {mars.vectors()[max_index]["datetime_str"]}")

### Question B.9

semi_major_axis_2 = 17.93
eccentricity_2 = 0.9679

halleys_orbital_coords = sp.Matrix([semi_major_axis_2*(sp.cos(E)-eccentricity_2), semi_major_axis_2*sp.sqrt(1-eccentricity_2**2)*sp.sin(E), 0])

halleys_rotation = rotation_composition.subs({omega:1.958, i:2.8308, Omega:1.031}) #used the orbital elements to transform coordinates from Halley's orbital plane
halleys_reference_coords = halleys_rotation*halleys_orbital_coords                 #to coordinates on our reference plane that is centred around the sun

days_over_time_period_3 = list(np.linspace(0.505842407619696, 1.0000134324800614, num=13704, endpoint=False)) #used increments of 1/T to work out (t-tau)/T for each day in our
days_over_time_period_3.extend(list(np.linspace(1.3432480061359442*(10**-5),0.16441256429109502, num=4559, endpoint=False))) #interval, (t-tau)/T = (0.505...) on day 0, (1.343...E-5)
halleys_eccentricity_anomalies = [eccentricity_anomaly(2*sp.Float(n)*sp.pi,eccentricity_2,verbose=False) for n in days_over_time_period_3] #on periapsis day, (0.164...) on the last day

halleys_approx_x_coords = [float(halleys_reference_coords.col(0)[0].subs({E:n}).evalf()) for n in halleys_eccentricity_anomalies] #calculates the approximate daily coordinates of
halleys_approx_y_coords = [float(halleys_reference_coords.col(0)[1].subs({E:n}).evalf()) for n in halleys_eccentricity_anomalies] #Halley's comet at midnight between 2025-01-01
halleys_approx_z_coords = [float(halleys_reference_coords.col(0)[2].subs({E:n}).evalf()) for n in halleys_eccentricity_anomalies] #and 2075-01-01 using the orbital elements

earth_2 = ast.Horizons(id="399",location="500@10", epochs={"start": "2025-01-01","stop": "2075-01-01","step": "1d"},id_type="id")

earth_predicted_x_coords = [float(row['x']) for row in earth_2.vectors()]
earth_predicted_y_coords = [float(row['y']) for row in earth_2.vectors()] #extracts the predicted daily coordinates of Earth at midnight
earth_predicted_z_coords = [float(row['z']) for row in earth_2.vectors()] #between 2025-01-01 and 2075-01-01 using the JPL Horizons database

fig = plt.figure(3, figsize=(8,6.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.dist = 1
ax.set_xlabel(r"X (AU)")
ax.set_ylabel(r"Y (AU)")
ax.set_zlabel(r"Z (AU)")
ax.plot(halleys_approx_x_coords, halleys_approx_y_coords, halleys_approx_z_coords, color="blue", label=r"Halley's comet approximate path")
ax.plot(earth_predicted_x_coords, earth_predicted_y_coords, earth_predicted_z_coords, color="green", label=r"Earth's predicted path")
ax.scatter(0, 0, 0, color='gold', s=50, label='Sun') 
ax.text(0, 0, 0, 'Sun', color='gold', fontsize=12)
ax.set_title(r"Paths of Halley's comet and Earth")
plt.legend() #plots the approximated postions of Halley's comet 
plt.show()   #and the predicted positions of Earth with reference to the Sun (that is at the origin) over the given time interval

### Question B.10

distances_2 = []

for n in range(len(halleys_approx_x_coords)): #creates a list of distances between the approximated postion of Halley's comet and the predicted positions of Earth each day in our time interval
    distances_2.append(np.sqrt((halleys_approx_x_coords[n]-earth_predicted_x_coords[n])**2+(halleys_approx_y_coords[n]-earth_predicted_y_coords[n])**2+(halleys_approx_z_coords[n]-earth_predicted_z_coords[n])**2)) 

days_of_interval = np.linspace(0,18263,num=18263,endpoint=False)

plt.figure(4)
plt.plot(days_of_interval,distances_2,color="blue")
plt.grid()
plt.xlabel(r"Days since 2025-01-01")
plt.ylabel(r"Distance between Earth and Halley's Comet (AU)")
plt.title(r"Distance between predicted positions of Earth and Halley's Comet against time")
plt.show() #plots the distance between the approximated postion of Halley's comet and the predicted positions of Earth for each day in our time interval

min_distance = min(distances_2)
min_index = distances_2.index(min_distance) #finds the min distance and the index of this min distance in our list of distances to help find the date this min distance occurs

print(f"Min distance between Earth's predicted path and Halley's approximated path {min_distance} AU")
print(f"Day of min distance between Earth's predicted path and Halley's approximated path is {earth_2.vectors()[min_index]["datetime_str"]}")

