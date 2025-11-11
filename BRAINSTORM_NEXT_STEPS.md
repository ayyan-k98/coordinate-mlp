# Brainstorming: Next Steps & Experiments

## 🎯 Current Status Summary

**What Works:**
- ✅ FCN architecture (548K parameters)
- ✅ Multi-scale training infrastructure (15, 20, 25, 30 grids)
- ✅ Curriculum learning framework (5 phases)
- ✅ Probabilistic coverage model
- ✅ Empty maps: ~99% coverage (overfit but shows capability)

**What Doesn't Work:**
- ❌ Value stability (Q-values exploding to infinity)
- ❌ Generalization (40% train-val gap)
- ❌ Structured environments (corridors 10-20%, caves 10-35%)
- ❌ Training completion (stopped at episode 450/1500)

**Key Insight:**
The agent can learn perfectly on simple maps but:
1. Values diverge due to reward scaling issues
2. Overfits to training maps
3. Struggles with complex structured obstacles

---

## 💡 IDEA CATEGORY A: Algorithm Changes

### A1. Switch to PPO (Proximal Policy Optimization)
**Why:** PPO is more stable than DQN for continuous/complex action spaces
- ✅ No Q-value divergence issues
- ✅ Better exploration through stochastic policy
- ✅ Trust region updates prevent catastrophic forgetting
- ❌ More complex to implement
- ❌ Requires policy network redesign

**Implementation Difficulty:** Medium (3-5 days)
**Expected Improvement:** +20-30% stability, +10% final performance

---

### A2. Prioritized Experience Replay (PER)
**Why:** Focus learning on "surprising" transitions
- ✅ Faster learning on difficult scenarios (corridors/caves)
- ✅ Better sample efficiency
- ❌ Adds computational overhead
- ❌ Can overfit to outliers if not tuned well

**Implementation Difficulty:** Low (1-2 days)
**Expected Improvement:** +5-10% sample efficiency

**Implementation:**
```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.priorities = np.zeros(capacity)
        self.alpha = alpha  # How much prioritization

    def sample(self, batch_size, beta=0.4):
        # Sample proportional to TD error
        probs = self.priorities ** self.alpha
        indices = np.random.choice(len(self), batch_size, p=probs)
        # Importance sampling weights
        weights = (len(self) * probs[indices]) ** (-beta)
        return batch, weights, indices

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.abs(td_errors) + 1e-6
```

---

### A3. Distributional DQN (C51 or QR-DQN)
**Why:** Learn distribution of returns instead of mean
- ✅ More robust to reward scale variations
- ✅ Better handles multi-modal returns (different map types)
- ✅ Natural uncertainty estimation
- ❌ Significant architecture changes needed
- ❌ Harder to debug

**Implementation Difficulty:** High (5-7 days)
**Expected Improvement:** +15-25% on complex maps

---

### A4. Soft Actor-Critic (SAC)
**Why:** State-of-the-art off-policy method
- ✅ Automatic entropy tuning (exploration)
- ✅ Very sample efficient
- ✅ Stable training
- ❌ Requires continuous action space (need action discretization)
- ❌ More hyperparameters

**Implementation Difficulty:** High (5-7 days)
**Expected Improvement:** +20-30% overall

---

## 💡 IDEA CATEGORY B: Architecture Improvements

### B1. Add Recurrent Layer (LSTM/GRU)
**Why:** Handle partial observability better
- ✅ Agent can remember where it's been
- ✅ Better long-term planning
- ✅ Natural for POMDP setting
- ❌ Slower training
- ❌ Needs sequence batching

**Implementation:**
```python
class RecurrentFCN(nn.Module):
    def __init__(self, ...):
        self.fcn = FCNCoverageNetwork(...)
        self.lstm = nn.LSTM(256, 256, num_layers=2)
        self.q_head = DuelingQNetwork(256, num_actions)

    def forward(self, state, hidden=None):
        spatial_features = self.fcn.get_features(state)  # [B, 256]
        lstm_out, hidden = self.lstm(spatial_features.unsqueeze(0), hidden)
        q_values = self.q_head(lstm_out.squeeze(0))
        return q_values, hidden
```

**Implementation Difficulty:** Medium (3-4 days)
**Expected Improvement:** +10-15% on complex maps

---

### B2. Multi-Head Architecture (Separate Heads per Map Type)
**Why:** Different strategies for different map types
- ✅ Prevents negative transfer between map types
- ✅ Explicit specialization
- ❌ More parameters
- ❌ Need map type as input (given by curriculum)

**Implementation:**
```python
class MultiHeadFCN(nn.Module):
    def __init__(self, ...):
        self.shared_encoder = FCNCoverageNetwork(...)
        self.heads = nn.ModuleDict({
            'empty': DuelingQNetwork(256, 9),
            'random': DuelingQNetwork(256, 9),
            'corridor': DuelingQNetwork(256, 9),
            'cave': DuelingQNetwork(256, 9),
        })

    def forward(self, state, map_type):
        features = self.shared_encoder.get_features(state)
        return self.heads[map_type](features)
```

**Implementation Difficulty:** Low (1-2 days)
**Expected Improvement:** +15-20% on structured maps

---

### B3. Attention Over History (Transformer)
**Why:** Attend to relevant past observations
- ✅ Better than LSTM for long sequences
- ✅ Parallelizable
- ❌ Memory intensive
- ❌ Needs positional encoding for time

**Implementation Difficulty:** High (4-5 days)
**Expected Improvement:** +10-20% on large maps

---

### B4. Auxiliary Tasks (Multi-Task Learning)
**Why:** Force network to learn useful representations
- ✅ Better feature learning
- ✅ Reduces overfitting
- ✅ Can predict: coverage map, obstacle map, frontier map

**Implementation:**
```python
class MultiTaskFCN(nn.Module):
    def forward(self, state):
        features = self.encoder(state)

        # Main task
        q_values = self.q_head(features)

        # Auxiliary tasks (only during training)
        coverage_pred = self.coverage_head(features)  # Predict next coverage
        obstacle_pred = self.obstacle_head(features)  # Predict obstacles
        frontier_pred = self.frontier_head(features)  # Predict frontiers

        return q_values, coverage_pred, obstacle_pred, frontier_pred

# Loss becomes:
total_loss = rl_loss + 0.1 * coverage_loss + 0.1 * obstacle_loss + 0.1 * frontier_loss
```

**Implementation Difficulty:** Medium (3-4 days)
**Expected Improvement:** +10-15% generalization

---

## 💡 IDEA CATEGORY C: Training Improvements

### C1. Curriculum Redesign
**Current Problem:** Phase transitions too abrupt

**Ideas:**
- **Gradual mixing:** Don't switch map types suddenly
  ```python
  # Instead of:
  Phase 1: 100% empty
  Phase 2: 100% random

  # Do:
  Phase 1: 100% empty
  Phase 1.5: 75% empty, 25% random
  Phase 2: 50% empty, 50% random
  Phase 2.5: 25% empty, 75% random
  Phase 3: 100% random
  ```

- **Performance gating:** Only advance when validation coverage > threshold
  ```python
  if validation_coverage > 0.7 and episodes_in_phase > min_episodes:
      advance_to_next_phase()
  ```

- **Dynamic difficulty:** Adjust obstacle density based on performance
  ```python
  if avg_coverage < 0.5:
      obstacle_density *= 0.9  # Make easier
  elif avg_coverage > 0.8:
      obstacle_density *= 1.1  # Make harder
  ```

**Implementation Difficulty:** Low (1-2 days)
**Expected Improvement:** +10-20% final performance

---

### C2. Data Augmentation
**Why:** Improve generalization with limited data
- Rotation (90°, 180°, 270°)
- Flipping (horizontal, vertical)
- Obstacle perturbation (add/remove small obstacles)

**Implementation:**
```python
def augment_state(state, aug_type):
    if aug_type == 'rot90':
        return np.rot90(state, k=1, axes=(1,2))
    elif aug_type == 'flip_h':
        return np.flip(state, axis=2)
    # etc.

# In store_transition:
if np.random.rand() < 0.5:
    state = augment_state(state, random_aug_type)
```

**Implementation Difficulty:** Low (1 day)
**Expected Improvement:** +5-10% generalization

---

### C3. Imitation Learning Bootstrap
**Why:** Give agent good starting policy
- Collect expert demonstrations using:
  - A* coverage path planning
  - Greedy frontier-based exploration
  - Or spiral/zigzag patterns

**Two-Phase Training:**
1. **Behavioral Cloning (BC):** Train on expert demos for 100 episodes
2. **RL Fine-tuning:** Continue with DQN

**Implementation:**
```python
# Phase 1: BC
for state, expert_action in expert_demos:
    action_probs = policy_net(state)
    loss = cross_entropy(action_probs, expert_action)
    loss.backward()

# Phase 2: DQN (normal)
```

**Implementation Difficulty:** Medium (3-4 days - need to generate demos)
**Expected Improvement:** +20-30% early performance, +10% final

---

### C4. Self-Play / Population-Based Training
**Why:** Maintain diversity, prevent overfitting
- Keep pool of 5-10 agents with different policies
- Sample opponents during training
- Gradually replace worst performers

**Implementation Difficulty:** High (5-7 days)
**Expected Improvement:** +15-25% robustness

---

## 💡 IDEA CATEGORY D: Representation Learning

### D1. Pre-train Encoder on Coverage Prediction
**Why:** Learn good spatial representations before RL
- Collect random trajectories
- Train encoder to predict:
  - Next coverage map
  - Final coverage map
  - Optimal next move (supervised)

**Implementation:**
```python
# Pre-training phase (100k steps)
for state, next_state in random_trajectories:
    encoded = encoder(state)
    predicted_coverage = decoder(encoded)
    loss = mse_loss(predicted_coverage, next_state.coverage)
    loss.backward()

# Then freeze encoder layers and train Q-head
```

**Implementation Difficulty:** Medium (3-4 days)
**Expected Improvement:** +15-20% sample efficiency

---

### D2. Contrastive Learning (SimCLR for States)
**Why:** Learn state representations that group similar states
- Similar coverage patterns → similar embeddings
- Different patterns → different embeddings

**Implementation:**
```python
# Create positive pairs (augmentations of same state)
state1 = augment(state)
state2 = augment(state)

# Create negative pairs (different states)
z1 = encoder(state1)
z2 = encoder(state2)

# Contrastive loss
loss = contrastive_loss(z1, z2, temperature=0.5)
```

**Implementation Difficulty:** Medium-High (4-5 days)
**Expected Improvement:** +10-15% generalization

---

## 💡 IDEA CATEGORY E: Exploration Strategies

### E1. Curiosity-Driven Exploration (ICM)
**Why:** Intrinsic motivation to visit novel states
- Add bonus reward for "surprising" transitions
- Surprise = prediction error of forward model

**Implementation:**
```python
class ICM(nn.Module):
    def __init__(self):
        self.forward_model = nn.Sequential(...)  # Predict next state
        self.inverse_model = nn.Sequential(...)  # Predict action from state transition

    def compute_intrinsic_reward(self, state, action, next_state):
        predicted_next = self.forward_model(state, action)
        surprise = mse(predicted_next, next_state)
        return surprise  # Higher surprise = higher intrinsic reward

# Total reward = extrinsic + beta * intrinsic
```

**Implementation Difficulty:** Medium (3-4 days)
**Expected Improvement:** +20-30% on complex maps

---

### E2. Count-Based Exploration
**Why:** Encourage visiting rare states
- Maintain count of state visits (using hash or NN)
- Bonus reward = 1/sqrt(count)

**Implementation:**
```python
state_counts = defaultdict(int)

def get_exploration_bonus(state):
    state_hash = hash(state.tobytes())
    count = state_counts[state_hash]
    state_counts[state_hash] += 1
    return 1.0 / (count + 1)**0.5

# Add to reward
total_reward = extrinsic_reward + 0.1 * exploration_bonus
```

**Implementation Difficulty:** Low (1-2 days)
**Expected Improvement:** +10-15% coverage on complex maps

---

### E3. Optimistic Initialization
**Why:** Encourage exploration through optimism
- Initialize Q-values to high values
- Agent tries everything to verify actual values

**Implementation:**
```python
# Instead of random init:
for param in q_network.parameters():
    if param.dim() == 2:  # Weights
        nn.init.constant_(param, 0.1)  # Small positive bias
    elif param.dim() == 1:  # Biases
        nn.init.constant_(param, 5.0)  # Large positive bias
```

**Implementation Difficulty:** Very Low (<1 day)
**Expected Improvement:** +5-10% early performance

---

## 💡 IDEA CATEGORY F: Reward Shaping

### F1. Potential-Based Shaping
**Why:** Guide agent without changing optimal policy
- Define potential function φ(s) (e.g., distance to nearest uncovered cell)
- Shaped reward: r' = r + γφ(s') - φ(s)
- This is **policy invariant** (doesn't change optimal policy)

**Implementation:**
```python
def potential(state):
    # Distance to nearest frontier
    frontiers = state.frontiers
    if len(frontiers) == 0:
        return 0.0
    agent_pos = state.agents[0].position
    min_dist = min(manhattan_distance(agent_pos, f) for f in frontiers)
    return -min_dist * 0.1  # Negative distance

shaped_reward = reward + gamma * potential(next_state) - potential(state)
```

**Implementation Difficulty:** Low (1-2 days)
**Expected Improvement:** +10-15% convergence speed

---

### F2. Hindsight Experience Replay (HER)
**Why:** Learn from failures by re-labeling goals
- After episode, take failed trajectories
- Pretend final state WAS the goal
- Store as successful trajectories

**Application to Coverage:**
```python
# Original episode: tried to cover everything, only got 60%
# Re-label: "goal was to cover these specific 60% of cells" → success!

for transition in episode:
    original_reward, done = compute_reward(transition, goal=full_coverage)
    # Store original
    replay_buffer.add(state, action, original_reward, next_state, done)

    # Also store with achieved goal
    achieved_coverage = final_state.coverage
    hindsight_reward, hindsight_done = compute_reward(transition, goal=achieved_coverage)
    replay_buffer.add(state, action, hindsight_reward, next_state, hindsight_done)
```

**Implementation Difficulty:** Medium (3-4 days)
**Expected Improvement:** +15-25% sample efficiency

---

## 💡 IDEA CATEGORY G: Debugging & Analysis Tools

### G1. Attention Visualization
**Why:** Understand what network focuses on
- Visualize spatial softmax attention weights
- Overlay on coverage map
- Check if attention correlates with frontiers

**Implementation:**
```python
# In FCN forward pass, save attention:
self.last_attention_weights = attention_weights

# Visualization:
import matplotlib.pyplot as plt
plt.imshow(state.coverage, alpha=0.5)
plt.imshow(attention_weights, alpha=0.5, cmap='hot')
plt.title(f"Attention at step {t}")
```

---

### G2. Q-Value Heatmaps
**Why:** See where agent thinks is valuable
- For each position on grid
- Visualize Q-values for "move to this position"

**Implementation:**
```python
def generate_q_heatmap(agent, state):
    H, W = state.coverage.shape
    q_map = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            # Simulate agent at (y, x)
            temp_state = state.copy()
            temp_state.agents[0].position = (y, x)
            q_values = agent.policy_net(temp_state)
            q_map[y, x] = q_values.max()

    plt.imshow(q_map, cmap='viridis')
    plt.title("Q-Value Heatmap")
```

---

### G3. Episode Replay Videos
**Why:** Manually inspect agent behavior
- Save episode frames
- Create video showing:
  - Agent path
  - Coverage evolution
  - Q-values over time
  - Attention weights

---

### G4. Failure Mode Analysis
**Why:** Understand systematic failures
- Collect episodes where coverage < 50%
- Cluster by failure type:
  - Stuck in loops
  - Ignoring areas
  - Inefficient paths
- Design targeted fixes

---

## 🎯 RECOMMENDED PRIORITY RANKING

### **Tier 1: Do First (High Impact, Low Effort)**
1. **Apply all fixes from TRAINING_FIXES.md** (1-2 days)
2. **C2: Data Augmentation** (1 day) - Easy generalization boost
3. **E2: Count-Based Exploration** (1 day) - Better coverage
4. **F1: Potential-Based Reward Shaping** (1 day) - Faster learning
5. **G1-G4: Debugging Tools** (2-3 days) - Critical for iteration

**Total: 1 week**
**Expected: Stable training, 60-70% validation coverage**

---

### **Tier 2: Do Second (High Impact, Medium Effort)**
6. **B2: Multi-Head Architecture** (2 days) - Handle diverse maps
7. **C1: Curriculum Redesign** (2 days) - Smoother learning
8. **A2: Prioritized Experience Replay** (2 days) - Sample efficiency
9. **C3: Imitation Learning Bootstrap** (3-4 days) - Jump-start training
10. **E1: Curiosity-Driven Exploration** (3-4 days) - Better exploration

**Total: 2 weeks**
**Expected: 70-80% validation coverage, completes all curriculum phases**

---

### **Tier 3: Experimental (High Impact, High Effort)**
11. **B1: Recurrent Layer** (3-4 days) - Long-term dependencies
12. **D1: Pre-train Encoder** (3-4 days) - Better representations
13. **A1: Switch to PPO** (5 days) - More stable algorithm
14. **A3: Distributional DQN** (5-7 days) - Handle uncertainty
15. **F2: Hindsight Experience Replay** (3-4 days) - Learn from failures

**Total: 3-4 weeks**
**Expected: 80-90% validation coverage, SOTA performance**

---

## 🚀 SUGGESTED EXPERIMENT SEQUENCE

### Week 1: Stabilization
- Day 1-2: Apply all critical fixes from TRAINING_FIXES.md
- Day 3: Add data augmentation
- Day 4: Add count-based exploration
- Day 5: Add potential-based shaping
- Day 6-7: Build debugging/visualization tools
- **Goal:** Stable training to episode 1000+

### Week 2: Architecture
- Day 1-2: Implement multi-head architecture
- Day 3-4: Redesign curriculum with gradual mixing
- Day 5-7: Add prioritized experience replay
- **Goal:** 70%+ validation coverage on all map types

### Week 3: Bootstrap & Exploration
- Day 1-3: Generate expert demos, implement imitation learning
- Day 4-7: Implement curiosity-driven exploration (ICM)
- **Goal:** 80%+ validation coverage

### Week 4: Advanced Methods
- Option A: Implement recurrent architecture
- Option B: Switch to PPO
- Option C: Pre-train encoder
- **Goal:** Compare approaches, pick best for final model

---

## 📊 SUCCESS METRICS

**By End of Week 1:**
- ✅ No gradient explosions
- ✅ Completes 1000 episodes without early stopping
- ✅ Validation coverage: 55-65%

**By End of Week 2:**
- ✅ Validation coverage: 70-75%
- ✅ All curriculum phases completed
- ✅ Corridor maps: >40% coverage

**By End of Week 3:**
- ✅ Validation coverage: 80-85%
- ✅ Corridor maps: >60% coverage
- ✅ Cave maps: >50% coverage

**By End of Week 4:**
- ✅ Validation coverage: 85-90%
- ✅ Best model generalizes to 40×40 maps
- ✅ Ready for deployment/publication

---

## 🤔 OPEN RESEARCH QUESTIONS

1. **Is FCN the right architecture?**
   - FCN is good for spatial reasoning
   - But no recurrence for memory
   - Consider hybrid: FCN + LSTM

2. **Is DQN the right algorithm?**
   - DQN struggles with exploration
   - PPO/SAC might be better
   - Or use DQN with strong exploration bonuses

3. **Is the reward structure correct?**
   - Currently: per-cell rewards
   - Alternative: global coverage percentage
   - Or: hierarchical rewards (area coverage + fine-grained)

4. **How to handle multi-agent extension?**
   - Current: single agent
   - Future: coordinated multi-agent
   - QMIX? MADDPG? Central critic?

5. **Can we learn map-agnostic policies?**
   - Current: different map types
   - Ideal: one policy for all maps
   - Need better representation learning

---

## 💭 WILD IDEAS (Low Priority, High Risk, High Reward)

### 1. Meta-Learning (MAML)
Learn to quickly adapt to new map types with few episodes.

### 2. World Models
Learn forward model of environment, plan in latent space.

### 3. Graph Neural Networks
Represent coverage space as graph, use GNN for reasoning.

### 4. Evolutionary Algorithms
Evolve population of policies instead of gradient descent.

### 5. Inverse RL
Learn reward function from expert demonstrations.

---

## 🎬 CONCLUSION

**Immediate Next Steps:**
1. Apply TRAINING_FIXES.md (1-2 days)
2. Run stability test (50 episodes)
3. If stable, proceed to Tier 1 improvements
4. Build debugging tools in parallel
5. Iterate based on what you learn

**Long-Term Vision:**
A robust, generalizable coverage planning agent that:
- Handles any map type
- Scales to large environments (50×50+)
- Achieves >85% coverage consistently
- Serves as foundation for multi-agent coordination

**Remember:** Start simple, fix stability first, then add complexity!
