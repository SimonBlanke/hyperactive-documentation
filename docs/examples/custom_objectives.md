# Custom Objectives

## Introduction

Custom objective functions are the heart of any optimization problem. This page demonstrates how to design and implement complex objective functions for various domains, from engineering optimization to business metric optimization and multi-objective scenarios.

## Mathematical Function Optimization

### Classic Benchmark Functions

```python
from hyperactive.base import BaseExperiment
from hyperactive.opt.gfo import BayesianOptimizer
import numpy as np
import math

class AckleyFunction(BaseExperiment):
    """Ackley function - multimodal optimization benchmark"""
    
    def __init__(self, dimensions=2):
        super().__init__()
        self.dimensions = dimensions
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        # Extract parameter vector
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        
        # Ackley function calculation
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        
        ackley = (-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - 
                  np.exp(sum_cos / n) + 20 + np.e)
        
        # Return negative for minimization (Hyperactive maximizes)
        return -ackley, {
            "function_value": ackley,
            "global_minimum": 0.0,  # Known global minimum
            "parameter_norm": np.linalg.norm(x)
        }

# Optimize Ackley function
experiment = AckleyFunction(dimensions=5)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Ackley function value:", -experiment.score(best_params)[0])
```

### Constrained Optimization

```python
class ConstrainedOptimization(BaseExperiment):
    """Optimization with constraints using penalty method"""
    
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        x, y, z = params["x"], params["y"], params["z"]
        
        # Objective function: maximize -(x² + y² + z²)
        objective = -(x**2 + y**2 + z**2)
        
        # Constraints
        constraint_violations = []
        
        # Constraint 1: x + y + z <= 1
        c1_violation = max(0, x + y + z - 1)
        constraint_violations.append(c1_violation)
        
        # Constraint 2: x² + y² >= 0.5
        c2_violation = max(0, 0.5 - (x**2 + y**2))
        constraint_violations.append(c2_violation)
        
        # Constraint 3: z >= 0
        c3_violation = max(0, -z)
        constraint_violations.append(c3_violation)
        
        # Penalty for constraint violations
        penalty_weight = 1000
        total_penalty = sum(constraint_violations) * penalty_weight
        
        # Penalized objective
        penalized_objective = objective - total_penalty
        
        return penalized_objective, {
            "raw_objective": objective,
            "constraint_violations": constraint_violations,
            "total_penalty": total_penalty,
            "is_feasible": sum(constraint_violations) == 0
        }

# Run constrained optimization
experiment = ConstrainedOptimization()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Best constrained parameters:", best_params)
print("Is feasible:", result[1]["is_feasible"])
print("Objective value:", result[1]["raw_objective"])
```

## Engineering Optimization

### Structural Design Optimization

```python
class BeamOptimization(BaseExperiment):
    """Optimize beam design for minimum weight with strength constraints"""
    
    def __init__(self, load=1000, length=3.0, max_stress=250e6, material_density=7850):
        super().__init__()
        self.load = load  # Applied load (N)
        self.length = length  # Beam length (m)
        self.max_stress = max_stress  # Maximum allowable stress (Pa)
        self.material_density = material_density  # Material density (kg/m³)
    
    def _paramnames(self):
        return ["width", "height", "thickness"]  # Rectangular hollow beam
    
    def _evaluate(self, params):
        width = max(0.01, params["width"])  # Minimum 1cm
        height = max(0.01, params["height"])  # Minimum 1cm
        thickness = max(0.001, min(width/2, height/2, params["thickness"]))  # Valid thickness
        
        try:
            # Calculate beam properties
            outer_area = width * height
            inner_area = (width - 2*thickness) * (height - 2*thickness)
            cross_sectional_area = outer_area - inner_area
            
            # Moment of inertia for rectangular hollow section
            I_outer = (width * height**3) / 12
            I_inner = ((width - 2*thickness) * (height - 2*thickness)**3) / 12
            moment_of_inertia = I_outer - I_inner
            
            # Calculate maximum stress (bending stress at midspan)
            max_moment = (self.load * self.length**2) / 8  # For simply supported beam
            max_stress = (max_moment * height/2) / moment_of_inertia
            
            # Calculate weight
            volume = cross_sectional_area * self.length
            weight = volume * self.material_density
            
            # Constraints
            stress_violation = max(0, max_stress - self.max_stress)
            
            # Objective: minimize weight (so maximize negative weight)
            if stress_violation > 0:
                # Heavy penalty for stress violation
                penalized_weight = weight + stress_violation * 1e6
            else:
                penalized_weight = weight
            
            return -penalized_weight, {
                "weight_kg": weight,
                "max_stress_Pa": max_stress,
                "stress_violation": stress_violation,
                "safety_factor": self.max_stress / max_stress if max_stress > 0 else float('inf'),
                "cross_sectional_area": cross_sectional_area,
                "moment_of_inertia": moment_of_inertia
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Optimize beam design
experiment = BeamOptimization(load=5000, length=4.0)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Optimal beam dimensions:", best_params)
print("Beam weight:", result[1]["weight_kg"], "kg")
print("Safety factor:", result[1]["safety_factor"])
```

### Control System Tuning

```python
class PIDControllerOptimization(BaseExperiment):
    """Optimize PID controller parameters for system response"""
    
    def __init__(self, setpoint=1.0, simulation_time=10.0):
        super().__init__()
        self.setpoint = setpoint
        self.simulation_time = simulation_time
    
    def _paramnames(self):
        return ["Kp", "Ki", "Kd"]  # PID gains
    
    def _evaluate(self, params):
        try:
            import scipy.signal as signal
            
            Kp = max(0, params["Kp"])
            Ki = max(0, params["Ki"])
            Kd = max(0, params["Kd"])
            
            # Define plant transfer function (second-order system)
            # G(s) = 1 / (s² + 2s + 1)
            plant_num = [1]
            plant_den = [1, 2, 1]
            plant = signal.TransferFunction(plant_num, plant_den)
            
            # PID controller transfer function
            # C(s) = Kp + Ki/s + Kd*s
            pid_num = [Kd, Kp, Ki]
            pid_den = [0, 1, 0]
            pid = signal.TransferFunction(pid_num, pid_den)
            
            # Closed-loop system
            open_loop = signal.series(pid, plant)
            closed_loop = signal.feedback(open_loop)
            
            # Simulate step response
            time = np.linspace(0, self.simulation_time, 1000)
            time, response = signal.step(closed_loop, T=time)
            
            # Performance metrics
            steady_state_error = abs(self.setpoint - response[-1])
            overshoot = max(0, (max(response) - self.setpoint) / self.setpoint * 100)
            
            # Rise time (10% to 90% of final value)
            final_value = response[-1]
            rise_start_idx = np.where(response >= 0.1 * final_value)[0][0]
            rise_end_idx = np.where(response >= 0.9 * final_value)[0][0]
            rise_time = time[rise_end_idx] - time[rise_start_idx]
            
            # Settling time (within 2% of final value)
            settling_band = 0.02 * abs(final_value)
            settled_indices = np.where(np.abs(response - final_value) <= settling_band)[0]
            settling_time = time[settled_indices[0]] if len(settled_indices) > 0 else self.simulation_time
            
            # Integral of absolute error
            error = self.setpoint - response
            iae = np.trapz(np.abs(error), time)
            
            # Combined performance score (minimize all metrics)
            performance_score = -(steady_state_error + 0.01*overshoot + 0.1*rise_time + 
                                0.1*settling_time + 0.001*iae)
            
            return performance_score, {
                "steady_state_error": steady_state_error,
                "overshoot_percent": overshoot,
                "rise_time": rise_time,
                "settling_time": settling_time,
                "iae": iae,
                "stable": np.all(np.abs(response) < 100)  # Stability check
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Optimize PID controller
experiment = PIDControllerOptimization()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Optimal PID gains:", best_params)
print("Performance metrics:", {k: v for k, v in result[1].items() if k != "stable"})
```

## Business Optimization

### Portfolio Optimization

```python
class PortfolioOptimization(BaseExperiment):
    """Optimize investment portfolio for risk-return trade-off"""
    
    def __init__(self, returns, risk_free_rate=0.02):
        super().__init__()
        self.returns = np.array(returns)  # Historical returns for each asset
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns)
    
    def _paramnames(self):
        return [f"weight_{i}" for i in range(self.n_assets)]
    
    def _evaluate(self, params):
        # Extract weights and normalize to sum to 1
        weights = np.array([params[f"weight_{i}"] for i in range(self.n_assets)])
        weights = np.abs(weights)  # Ensure positive weights
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(self.returns.T, weights)
        
        # Portfolio statistics
        expected_return = np.mean(portfolio_returns)
        portfolio_risk = np.std(portfolio_returns)
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Multi-objective score: maximize Sharpe ratio, minimize max drawdown
        score = sharpe_ratio + 0.5 * (-max_drawdown)  # Combine objectives
        
        return score, {
            "expected_return": expected_return,
            "portfolio_risk": portfolio_risk,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "weights": weights.tolist(),
            "diversification_ratio": 1 / np.sum(weights**2)  # Higher is more diversified
        }

# Generate sample return data
np.random.seed(42)
n_assets = 5
n_periods = 252  # Trading days in a year
returns = np.random.normal(0.08/252, 0.2/np.sqrt(252), (n_assets, n_periods))

# Add some correlation structure
correlation_matrix = np.array([
    [1.0, 0.6, 0.3, 0.2, 0.1],
    [0.6, 1.0, 0.4, 0.2, 0.15],
    [0.3, 0.4, 1.0, 0.5, 0.2],
    [0.2, 0.2, 0.5, 1.0, 0.3],
    [0.1, 0.15, 0.2, 0.3, 1.0]
])

# Apply correlation
returns = np.dot(np.linalg.cholesky(correlation_matrix), returns)

# Optimize portfolio
experiment = PortfolioOptimization(returns)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Optimal portfolio weights:", result[1]["weights"])
print("Expected return:", f"{result[1]['expected_return']*252:.1%}")
print("Portfolio risk:", f"{result[1]['portfolio_risk']*np.sqrt(252):.1%}")
print("Sharpe ratio:", f"{result[1]['sharpe_ratio']:.3f}")
```

### Supply Chain Optimization

```python
class SupplyChainOptimization(BaseExperiment):
    """Optimize supply chain network configuration"""
    
    def __init__(self, demand_locations, potential_facilities, transport_costs):
        super().__init__()
        self.demand_locations = demand_locations  # List of (x, y, demand)
        self.potential_facilities = potential_facilities  # List of (x, y, capacity, fixed_cost)
        self.transport_costs = transport_costs  # Cost per unit per distance
        self.n_facilities = len(potential_facilities)
    
    def _paramnames(self):
        # Binary variables for facility selection + capacity utilization
        return ([f"facility_{i}" for i in range(self.n_facilities)] + 
                [f"capacity_util_{i}" for i in range(self.n_facilities)])
    
    def _evaluate(self, params):
        try:
            # Extract facility decisions (binary: open or not)
            facility_decisions = [params[f"facility_{i}"] > 0.5 for i in range(self.n_facilities)]
            capacity_utils = [max(0, min(1, params[f"capacity_util_{i}"])) for i in range(self.n_facilities)]
            
            if not any(facility_decisions):
                return float('-inf'), {"error": "no_facilities_selected"}
            
            # Calculate costs
            fixed_costs = sum(self.potential_facilities[i][3] for i in range(self.n_facilities) 
                            if facility_decisions[i])
            
            # Calculate available capacity at each facility
            available_capacity = [
                self.potential_facilities[i][2] * capacity_utils[i] if facility_decisions[i] else 0
                for i in range(self.n_facilities)
            ]
            
            # Assign demand to facilities (simple greedy assignment)
            total_transport_cost = 0
            unmet_demand = 0
            
            for demand_loc in self.demand_locations:
                demand_x, demand_y, demand = demand_loc
                remaining_demand = demand
                
                # Calculate distances to all open facilities
                facility_distances = []
                for i, facility in enumerate(self.potential_facilities):
                    if facility_decisions[i] and available_capacity[i] > 0:
                        facility_x, facility_y, capacity, _ = facility
                        distance = np.sqrt((demand_x - facility_x)**2 + (demand_y - facility_y)**2)
                        facility_distances.append((i, distance))
                
                # Sort by distance and assign demand
                facility_distances.sort(key=lambda x: x[1])
                
                for facility_idx, distance in facility_distances:
                    if remaining_demand <= 0:
                        break
                    
                    supply_amount = min(remaining_demand, available_capacity[facility_idx])
                    transport_cost = supply_amount * distance * self.transport_costs
                    total_transport_cost += transport_cost
                    
                    available_capacity[facility_idx] -= supply_amount
                    remaining_demand -= supply_amount
                
                unmet_demand += remaining_demand
            
            # Penalty for unmet demand
            unmet_demand_penalty = unmet_demand * 1000
            
            # Total cost (minimize)
            total_cost = fixed_costs + total_transport_cost + unmet_demand_penalty
            
            return -total_cost, {
                "fixed_costs": fixed_costs,
                "transport_costs": total_transport_cost,
                "unmet_demand": unmet_demand,
                "total_cost": total_cost,
                "facilities_opened": sum(facility_decisions),
                "capacity_utilization": np.mean([capacity_utils[i] for i in range(self.n_facilities) 
                                               if facility_decisions[i]])
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Define supply chain problem
demand_locations = [
    (10, 20, 100),  # (x, y, demand)
    (30, 40, 150),
    (50, 10, 80),
    (20, 60, 120)
]

potential_facilities = [
    (15, 25, 200, 5000),  # (x, y, capacity, fixed_cost)
    (35, 35, 250, 6000),
    (45, 15, 180, 4500),
    (25, 55, 220, 5500)
]

transport_costs = 2.0  # Cost per unit per distance

# Optimize supply chain
experiment = SupplyChainOptimization(demand_locations, potential_facilities, transport_costs)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Supply chain optimization results:")
print("Total cost:", result[1]["total_cost"])
print("Facilities opened:", result[1]["facilities_opened"])
print("Unmet demand:", result[1]["unmet_demand"])
```

## Multi-Objective Optimization

### Pareto Front Optimization

```python
class MultiObjectiveExperiment(BaseExperiment):
    """Multi-objective optimization with Pareto front analysis"""
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        
        # Objective 1: Minimize distance from origin
        obj1 = np.sqrt(x**2 + y**2)
        
        # Objective 2: Minimize distance from point (1, 1)
        obj2 = np.sqrt((x-1)**2 + (y-1)**2)
        
        # Objective 3: Maximize x + y (so minimize -(x + y))
        obj3 = -(x + y)
        
        # For multi-objective, return tuple of objectives
        # Note: Some optimizers can handle multi-objective directly
        return (obj1, obj2, obj3), {
            "objective_1": obj1,
            "objective_2": obj2,
            "objective_3": obj3,
            "dominated_by_origin": obj1 < 0.5 and obj2 > 0.5,
            "dominated_by_target": obj1 > 0.5 and obj2 < 0.5
        }

# For demo, we'll use weighted sum approach
class WeightedMultiObjective(BaseExperiment):
    """Convert multi-objective to single objective using weights"""
    
    def __init__(self, weights=[0.4, 0.4, 0.2]):
        super().__init__()
        self.weights = weights
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        
        # Same objectives as above
        obj1 = np.sqrt(x**2 + y**2)
        obj2 = np.sqrt((x-1)**2 + (y-1)**2)
        obj3 = -(x + y)
        
        # Normalize objectives (approximate)
        obj1_norm = obj1 / 2.0  # Assuming max distance ~2
        obj2_norm = obj2 / 2.0
        obj3_norm = (obj3 + 2) / 4  # Shift and scale
        
        # Weighted sum (minimize all, so negate for maximization)
        weighted_sum = -(self.weights[0] * obj1_norm + 
                        self.weights[1] * obj2_norm + 
                        self.weights[2] * obj3_norm)
        
        return weighted_sum, {
            "obj1_distance_origin": obj1,
            "obj2_distance_target": obj2,
            "obj3_sum": -obj3,
            "weighted_objectives": [obj1_norm, obj2_norm, obj3_norm]
        }

# Test different weight combinations
weight_combinations = [
    [0.6, 0.3, 0.1],  # Emphasize objective 1
    [0.3, 0.6, 0.1],  # Emphasize objective 2
    [0.1, 0.3, 0.6],  # Emphasize objective 3
    [0.33, 0.33, 0.34]  # Equal weights
]

pareto_solutions = []

for i, weights in enumerate(weight_combinations):
    experiment = WeightedMultiObjective(weights)
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    
    result = experiment.score(best_params)
    pareto_solutions.append({
        "weights": weights,
        "parameters": best_params,
        "objectives": [result[1]["obj1_distance_origin"], 
                      result[1]["obj2_distance_target"], 
                      result[1]["obj3_sum"]]
    })
    
    print(f"Weight combination {i+1}: {weights}")
    print(f"Solution: {best_params}")
    print(f"Objectives: {pareto_solutions[-1]['objectives']}")
    print()

print("Pareto front approximation:")
for i, sol in enumerate(pareto_solutions):
    print(f"Solution {i+1}: Objectives = {[f'{obj:.3f}' for obj in sol['objectives']]}")
```

## Simulation-Based Optimization

### Monte Carlo Optimization

```python
class MonteCarloExperiment(BaseExperiment):
    """Optimize parameters of a stochastic simulation"""
    
    def __init__(self, n_simulations=1000):
        super().__init__()
        self.n_simulations = n_simulations
    
    def _paramnames(self):
        return ["strategy_aggressiveness", "risk_threshold", "rebalance_frequency"]
    
    def _evaluate(self, params):
        try:
            aggressiveness = max(0, min(1, params["strategy_aggressiveness"]))
            risk_threshold = max(0, params["risk_threshold"])
            rebalance_freq = max(1, int(params["rebalance_frequency"]))
            
            # Run Monte Carlo simulation
            simulation_results = []
            
            for _ in range(self.n_simulations):
                result = self._simulate_strategy(aggressiveness, risk_threshold, rebalance_freq)
                simulation_results.append(result)
            
            # Calculate statistics
            mean_return = np.mean(simulation_results)
            std_return = np.std(simulation_results)
            downside_risk = np.std([r for r in simulation_results if r < 0])
            probability_loss = np.mean([1 for r in simulation_results if r < 0])
            var_95 = np.percentile(simulation_results, 5)  # Value at Risk
            
            # Risk-adjusted return score
            sharpe_like_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Combined score considering return and risk
            score = mean_return - 0.5 * downside_risk - 10 * probability_loss
            
            return score, {
                "mean_return": mean_return,
                "volatility": std_return,
                "downside_risk": downside_risk,
                "probability_loss": probability_loss,
                "var_95": var_95,
                "sharpe_like": sharpe_like_ratio
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}
    
    def _simulate_strategy(self, aggressiveness, risk_threshold, rebalance_freq):
        """Simulate a single run of the trading strategy"""
        np.random.seed(None)  # Different seed for each simulation
        
        # Simulate market returns (random walk with drift)
        n_periods = 252  # One year of trading days
        daily_returns = np.random.normal(0.0005, 0.02, n_periods)  # ~12.5% annual return, 32% volatility
        
        portfolio_value = 1.0  # Start with $1
        cash_position = 0.0
        
        for day in range(n_periods):
            # Determine position based on aggressiveness
            market_exposure = aggressiveness
            
            # Risk management
            if portfolio_value < (1 - risk_threshold):
                market_exposure *= 0.5  # Reduce exposure when losses exceed threshold
            
            # Rebalancing
            if day % rebalance_freq == 0:
                # Rebalance to target allocation
                pass
            
            # Apply daily return
            daily_return = daily_returns[day] * market_exposure
            portfolio_value *= (1 + daily_return)
            
            # Stop loss
            if portfolio_value < 0.5:  # 50% stop loss
                break
        
        # Return final performance
        return portfolio_value - 1.0  # Net return

# Optimize Monte Carlo strategy
experiment = MonteCarloExperiment(n_simulations=500)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Optimal strategy parameters:", best_params)
print("Expected return:", f"{result[1]['mean_return']:.1%}")
print("Volatility:", f"{result[1]['volatility']:.1%}")
print("Probability of loss:", f"{result[1]['probability_loss']:.1%}")
print("VaR (95%):", f"{result[1]['var_95']:.1%}")
```

## Custom Metrics and Scoring

### Complex Business Metrics

```python
class BusinessMetricOptimization(BaseExperiment):
    """Optimize for complex business KPIs"""
    
    def __init__(self, customer_data, cost_structure):
        super().__init__()
        self.customer_data = customer_data
        self.cost_structure = cost_structure
    
    def _paramnames(self):
        return ["price", "marketing_spend", "service_level", "product_quality"]
    
    def _evaluate(self, params):
        price = max(10, params["price"])  # Minimum price
        marketing_spend = max(0, params["marketing_spend"])
        service_level = max(0, min(1, params["service_level"]))  # 0-1 scale
        quality = max(0, min(1, params["product_quality"]))  # 0-1 scale
        
        # Customer acquisition model
        demand = self._calculate_demand(price, marketing_spend, service_level, quality)
        
        # Revenue calculation
        revenue = price * demand
        
        # Cost calculation
        variable_costs = demand * self.cost_structure["unit_cost"]
        fixed_costs = self.cost_structure["fixed_cost"]
        marketing_costs = marketing_spend
        service_costs = service_level * self.cost_structure["service_cost_per_level"]
        quality_costs = quality * self.cost_structure["quality_cost_per_level"]
        
        total_costs = variable_costs + fixed_costs + marketing_costs + service_costs + quality_costs
        
        # Profit
        profit = revenue - total_costs
        
        # Customer satisfaction and retention
        satisfaction = self._calculate_satisfaction(price, service_level, quality)
        retention_rate = 0.5 + 0.4 * satisfaction  # 50-90% retention based on satisfaction
        
        # Long-term customer value
        customer_lifetime_periods = 1 / (1 - retention_rate) if retention_rate < 1 else 10
        customer_lifetime_value = customer_lifetime_periods * profit / demand if demand > 0 else 0
        
        # Market share estimate
        competitor_price = 50  # Assumed competitor price
        price_competitiveness = max(0, 1 - (price - competitor_price) / competitor_price)
        market_share = 0.1 + 0.3 * price_competitiveness + 0.2 * quality + 0.1 * service_level
        market_share = min(0.8, market_share)  # Cap at 80%
        
        # Combined business score
        business_score = (0.4 * profit + 
                         0.3 * customer_lifetime_value + 
                         0.2 * satisfaction * 100 + 
                         0.1 * market_share * 100)
        
        return business_score, {
            "revenue": revenue,
            "profit": profit,
            "demand": demand,
            "total_costs": total_costs,
            "customer_satisfaction": satisfaction,
            "retention_rate": retention_rate,
            "customer_lifetime_value": customer_lifetime_value,
            "market_share": market_share,
            "profit_margin": profit / revenue if revenue > 0 else 0
        }
    
    def _calculate_demand(self, price, marketing_spend, service_level, quality):
        """Calculate demand based on price and other factors"""
        # Price elasticity
        base_demand = 1000
        price_effect = base_demand * np.exp(-0.02 * price)
        
        # Marketing effect (diminishing returns)
        marketing_effect = 1 + 0.5 * np.log(1 + marketing_spend / 1000)
        
        # Quality and service effects
        quality_effect = 1 + 0.3 * quality
        service_effect = 1 + 0.2 * service_level
        
        total_demand = price_effect * marketing_effect * quality_effect * service_effect
        return max(0, total_demand)
    
    def _calculate_satisfaction(self, price, service_level, quality):
        """Calculate customer satisfaction"""
        # Value perception (quality vs price)
        value_score = quality / (price / 50)  # Normalized by reference price
        
        # Service satisfaction
        service_score = service_level
        
        # Overall satisfaction
        satisfaction = 0.6 * min(1, value_score) + 0.4 * service_score
        return max(0, min(1, satisfaction))

# Define business parameters
customer_data = {"segments": ["premium", "standard", "budget"]}
cost_structure = {
    "unit_cost": 20,
    "fixed_cost": 10000,
    "service_cost_per_level": 5000,
    "quality_cost_per_level": 8000
}

# Optimize business metrics
experiment = BusinessMetricOptimization(customer_data, cost_structure)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

result = experiment.score(best_params)
print("Optimal business parameters:", best_params)
print("Business metrics:")
for metric, value in result[1].items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.2f}")
```

## Best Practices for Custom Objectives

### Objective Function Design Checklist

```python
class BestPracticeExperiment(BaseExperiment):
    """Template demonstrating best practices for objective functions"""
    
    def __init__(self):
        super().__init__()
        self.evaluation_count = 0
        self.best_score = float('-inf')
    
    def _paramnames(self):
        # 1. Use descriptive parameter names
        return ["learning_rate", "regularization", "batch_size"]
    
    def _evaluate(self, params):
        self.evaluation_count += 1
        
        try:
            # 2. Validate input parameters
            lr = max(1e-6, min(1.0, params["learning_rate"]))
            reg = max(0, params["regularization"])
            batch_size = max(1, int(params["batch_size"]))
            
            # 3. Handle edge cases gracefully
            if lr < 1e-5:
                return float('-inf'), {"error": "learning_rate_too_small"}
            
            # 4. Implement your core logic
            score = self._compute_objective(lr, reg, batch_size)
            
            # 5. Track progress
            if score > self.best_score:
                self.best_score = score
                print(f"New best score: {score:.6f}")
            
            # 6. Return comprehensive metadata
            metadata = {
                "learning_rate_used": lr,
                "regularization_used": reg,
                "batch_size_used": batch_size,
                "evaluation_number": self.evaluation_count,
                "convergence_metric": score * 0.9,  # Example derived metric
                "parameter_norm": np.sqrt(lr**2 + reg**2 + (batch_size/100)**2)
            }
            
            return score, metadata
            
        except Exception as e:
            # 7. Robust error handling
            return float('-inf'), {
                "error": str(e),
                "evaluation_number": self.evaluation_count
            }
    
    def _compute_objective(self, lr, reg, batch_size):
        """Implement your domain-specific objective here"""
        # Example: penalize extreme values
        lr_penalty = -abs(np.log10(lr) + 3)**2  # Prefer lr around 1e-3
        reg_penalty = -reg**2  # Prefer smaller regularization
        batch_penalty = -abs(batch_size - 64)**2 / 1000  # Prefer batch size around 64
        
        return lr_penalty + reg_penalty + batch_penalty

# Example usage
experiment = BestPracticeExperiment()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best practice optimization result:", best_params)
print("Final evaluation count:", experiment.evaluation_count)
```

## References

- Mathematical optimization benchmarks: [https://en.wikipedia.org/wiki/Test_functions_for_optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
- Multi-objective optimization: [https://en.wikipedia.org/wiki/Multi-objective_optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization)
- Engineering design optimization principles
- Business metrics and KPI optimization strategies