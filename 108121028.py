import random
import copy
import math
import matplotlib.pyplot as plt

class Algorithm_Simulated_Annealing:

    def __init__(self, puzzle, iterations, length, final_state=[['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]):
        self.puzzle = puzzle
        self.temperature = 1000
        self.cooling_rate = 0.995
        self.iterations = iterations
        self.frontier_length = length
        self.ans = final_state
    
    def location_mapping(self):
        self.loc_info = {}
        for i in self.ans:
            for j in i:
                self.loc_info[j] = (self.ans.index(i), i.index(j))
    
    def heuristic_value(self, puzzle):
        heuristic = 0
        for i in range(len(puzzle)):
            for j in range(len(puzzle[i])):
                tile = puzzle[i][j]
                if(tile!=self.ans[i][j]):
                    heuristic+=1
                if(tile != '0'):
                    goal_x, goal_y = self.loc_info[tile]
                    heuristic += abs(i - goal_x) + abs(j - goal_y)
        return heuristic
    
    def display_puzzle_stages(self, puzzle):
        puzzle_string = "\n"
        for i in puzzle:
            puzzle_string += '['+' '.join(i) +']' "\n"
        puzzle_string += "\n"
        return puzzle_string

    def shift(self, puzzle, visited):
        self.temperature = self.temperature * self.cooling_rate  # Cooling schedule (Decay)
        pos = [(i, j) for i in range(len(puzzle)) for j in range(len(puzzle[i])) if puzzle[i][j] == '0'][0]
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Down, Up, Right, Left
        for direction in directions:
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
            if 0 <= new_pos[0] < len(puzzle) and 0 <= new_pos[1] < len(puzzle[0]):
                new_puzzle = copy.deepcopy(puzzle)
                new_puzzle[pos[0]][pos[1]], new_puzzle[new_pos[0]][new_pos[1]] = new_puzzle[new_pos[0]][new_pos[1]], new_puzzle[pos[0]][pos[1]]
                neighbors.append(new_puzzle)
        curr_heuristic = self.heuristic_value(puzzle)
        heuristic_list = [self.heuristic_value(n) for n in neighbors]
        best_move = None
        control = True
        while (control==True):
            new_puzzle_index = random.randint(0, len(heuristic_list) - 1)
            new_heuristic = heuristic_list[new_puzzle_index]
            energy_diff = new_heuristic - curr_heuristic
            if(len(visited)==self.frontier_length):
                visited=visited[::-1]
                visited.pop()
                visited=visited[::-1]
            if energy_diff <= 0 or random.uniform(0, 1) >= math.exp(-energy_diff / self.temperature):
                best_move = neighbors[new_puzzle_index]
                if(neighbors[new_puzzle_index] in visited):
                    continue
                visited.append(neighbors[new_puzzle_index])
                break
        if best_move is None:  # If no better state was found, pick one randomly from the neighbor
            best_move = neighbors[random.randint(0, len(neighbors) - 1)]

        return best_move

    def display(self, puzzle, visited):
        self.location_mapping()
        current_puzzle = puzzle
        count=0
        while current_puzzle != self.ans and count<self.iterations:
            next_puzzle = self.shift(current_puzzle, visited)
            count+=1
            if next_puzzle == self.ans:
                print("Puzzle Solved -")
                print(self.display_puzzle_stages(next_puzzle))
                return
            current_puzzle = next_puzzle
            print(f"Step {count} -\n")
            print(self.display_puzzle_stages(current_puzzle))
        print("Partial Solution Reached -")
        print(self.display_puzzle_stages(current_puzzle))





class Genetic_Algorithm:

    def __init__(self, puzzle, generations, final_state=[['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]):
        self.puzzle = puzzle
        self.generations = generations
        self.ans = final_state
        self.ans_code = "111111111"
        self.code_length = len(self.ans)*len(self.ans[0])
    
    def location_mapping(self):
        self.loc_info = {}
        for i in self.ans:
            for j in i:
                self.loc_info[j] = (self.ans.index(i), i.index(j))
    
    def puzzle_encoder(self):
        puzzle_code = ""
        for i in range(len(self.puzzle)):
            for j in range(len(self.puzzle[i])):
                tile = self.puzzle[i][j]
                if(tile!=self.ans[i][j]):
                    puzzle_code+='0'
                else:
                    puzzle_code+='1'
        self.population_size = puzzle_code.count('0')
        if(self.population_size%2 != 0):
            self.population_size += 1
        return puzzle_code
    
    def population_creation(self):
        population_list = []
        count = 0
        self.puzzle_encoder()
        while(count<self.population_size):
            temp_code = ""
            for j in range(self.code_length):
                temp_code+=str(random.randint(0, 1))
            if(temp_code in population_list or temp_code.count('0')<2):
                continue
            population_list.append(temp_code)
            count+=1
        return population_list
    
    def fitness_score_mapping(self, population_list):
        population = population_list
        score_mapping = []
        for states in population:
            score_mapping.append((states, states.count('1')))
        return score_mapping
    
    def selection(self,parent_1, parent_2, score_1, score_2):
        parent=parent_1
        if(score_1>score_2):
            if(random.randint(1,10)<=2):
                parent=parent_2
        else:
            if(random.randint(1,10)>2):
                parent=parent_2
        return parent
            
    def selection_and_crossover(self, score, crossed):
        score_mapping = score
        while(True):
            parent_1, score_1 = score_mapping[random.randint(0,len(score_mapping)-1)]
            parent_2, score_2 = score_mapping[random.randint(0,len(score_mapping)-1)]
            parent_3, score_3 = score_mapping[random.randint(0,len(score_mapping)-1)]
            parent_4, score_4 = score_mapping[random.randint(0,len(score_mapping)-1)]
            #Elitisim
            parent_1 = self.selection(parent_1,parent_2,score_1,score_2)
            parent_2 = self.selection(parent_3,parent_4,score_3,score_4)
            if((parent_1, parent_2) in crossed):
                continue
            else:
                crossed.append((parent_1, parent_2))
                break
        # 50% Crossover
        pre = self.code_length//2
        post = (self.code_length-self.code_length//2)
        child_1 = parent_1[:pre] + parent_2[post-1:]
        child_2 = parent_2[:pre] + parent_1[post-1:]
        mutated_child_1 = child_1
        mutated_child_2 = child_2
        mutation_chance = random.randint(1,100)
        if(mutation_chance == 1):
            mutation_index_child_1, mutation_index_child_2 = random.randint(0,len(child_1)-1), random.randint(0,len(child_2)-1)
            mutated_child_1 = child_1[:mutation_index_child_1]+str(random.randint(0,1))+child_1[mutation_index_child_1+1:]
            mutated_child_2 = child_2[:mutation_index_child_2]+str(random.randint(0,1))+child_2[mutation_index_child_2+1:]

        return mutated_child_1, mutated_child_2

    def generation(self, crossed):
        initial_population_list = self.population_creation()
        score_mapping = self.fitness_score_mapping(initial_population_list)
        current_generation = 0
        crossed_list = crossed
        #Statistics
        x_generation = []
        y_score = []

        while(current_generation<=self.generations):

            max_score=0
            for score in score_mapping:
                if(score[1]>=max_score):
                    max_score=score[1]
            x_generation.append(current_generation+1)
            y_score.append(max_score)

            future_population = []
            while(len(future_population)<self.population_size):
                children = self.selection_and_crossover(score_mapping, crossed_list)
                future_population.append(children[0])
                future_population.append(children[1])
            score_mapping = self.fitness_score_mapping(future_population)
            if(self.ans_code in future_population):
                # print(future_population)
                print(f"\nSolution Found in {current_generation} Generation\n")
                print("\nCurrent Population\n")
                print(self.display_puzzle_stages(future_population))
                x_generation.append(current_generation+1)
                y_score.append(self.code_length)
                return x_generation, y_score
            crossed_list = []
            current_generation += 1
        print(f"\nPartial Solution Found and Current Generation {self.generations}\n")
        print("\nCurrent Population\n")
        print(self.display_puzzle_stages(future_population))
        return x_generation, y_score

    def display_puzzle_stages(self, population):
        puzzle_string = ""
        misplaced_tiles = []
        for states in range(len(population)):
            temp = []
            for bit in range(len(population[states])):
                if(population[states][bit]=='1'):
                    puzzle_string += str(bit)+' '
                else:
                    puzzle_string += 'x '
                    temp.append(str(bit))
            misplaced_tiles.append(temp)
            puzzle_string=puzzle_string.rstrip()
            puzzle_string+='\n\n'

        puzzle_string=puzzle_string.rstrip()
        puzzle_states = puzzle_string.split("\n\n")
        for i,states in enumerate(puzzle_states):
            for j,tile in enumerate(states):
                if(tile=='x'):
                    if(len(misplaced_tiles[i])>1):
                        missing_tile = misplaced_tiles[i].pop(random.randint(0,len(misplaced_tiles[i])-1))
                    else:
                        missing_tile = misplaced_tiles[i].pop()
                    puzzle_states[i] = puzzle_states[i][:j]+missing_tile+puzzle_states[i][j+1:]

        puzzle_string = ""
        for state in puzzle_states:
            start_index = 0
            splice = len(self.ans)*2-1
            for row in range(len(self.ans)):
                puzzle_string+=f"[{state[start_index:splice]}]\n"
                start_index=splice+1
                splice=splice+start_index
            puzzle_string+="\n"
        return puzzle_string
    
# Example usage for Simulated Annealing
def call_simulated_annealing():
    puzzle = [['4', '5', '0'], ['1', '2', '7'], ['6', '8', '3']] 
    visited = []
    iterations = 1000
    frontier_length = 10
    print("\nInput Puzzle -")
    instance = Algorithm_Simulated_Annealing(puzzle,iterations,frontier_length)
    print(instance.display_puzzle_stages(puzzle))
    instance.display(puzzle,visited)
    return

# Example usage for Genetic Algorithm
def call_genetic_algorithm():
    puzzle = [['8', '7', '6'], ['3', '4', '5'], ['0', '1', '2']] 
    generations = 1000
    crossed = []
    instance = Genetic_Algorithm(puzzle,generations)
    x_generation, y_score=instance.generation(crossed)
    plt.plot(x_generation, y_score)
    plt.xlim(0, max(x_generation)+2)
    plt.ylim(min(y_score), max(y_score)+2)
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.title('Fitness Score v/s Generations')
    plt.show()

#Using Simulated Annealing Algorithm
call_simulated_annealing()
#Using Genetic Alogrithm
call_genetic_algorithm()