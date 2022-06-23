from django.shortcuts import render
from .forms import limit, packValues
from .models import pack
import numpy as np




# Create your views here.


class GA:
    PopulationSize = 50
    def __init__(self,WeightLimit,MutationProbability):
        self.WeightLimit = WeightLimit
        self.MutationProbability = MutationProbability


    def ReturnWeightValue(self,PacksList, Packs):  # return weight and value for individuals 
        weight = 0
        value = 0
        for o in PacksList:
            o = Packs[o]
            weight += o[0]
            value += o[1]
        return weight, value

    def CheckWeight(self,PacksList, WeightLimit, Packs):   # check if the generated population exeeds the weight limit
        pack = []
        for p in PacksList:
            pack.append(p)
            weight, value = self.ReturnWeightValue(pack, Packs)
            if weight > WeightLimit:
                pack.pop()
                return pack
        return pack

    def InitialPopulation(self,PacksList, PopulationSize, WeightLimit, Packs):    #generate the first population Randomly
        Population = []
        numberOfPacks = len(Packs)
        for i in range(PopulationSize):         
            sol_i = PacksList[np.random.choice(list(range(numberOfPacks)), numberOfPacks, replace=False)]
            sol_i = self.CheckWeight(sol_i, WeightLimit, Packs)
            Population.append(sol_i)
        return np.array(Population)


    def Fitness(self,population, Packs):         # calculate fitness for all individuals using ReturnWeightValue()
        FitnessList = np.zeros(self.PopulationSize)
        for i in  range(self.PopulationSize):     
            _ , FitnessList[i] = self.ReturnWeightValue(population[i], Packs)  
        return FitnessList


    def RouletteSelection(population,FitnessList): # selecting two lists of individuals as parents to pair them for crossover
        TotalFitness = FitnessList.sum()
        prob_list = FitnessList/TotalFitness
        
        SelectedParents_1 = np.random.choice(list(range(len(population))), len(population),p=prob_list, replace=True)
        SelectedParents_2 = np.random.choice(list(range(len(population))), len(population),p=prob_list, replace=True)
        
        SelectedParents_1 = population[SelectedParents_1]
        SelectedParents_2 = population[SelectedParents_2]
        
        return np.array([SelectedParents_1,SelectedParents_2])


    def CrossOver(self,parent_1, parent_2, WeightLimit, Packs):  # crossover each pair in parents list
        offspring = []
        for i in zip(parent_1, parent_2):
            offspring.extend(i)
            offspring = self.CheckWeight(offspring, WeightLimit, Packs)
        return offspring   

    def CrossoverPairs(self,ParentsList, WeightLimit, Packs):   # crossovering parents using CrossOver() to create a new popualtion
        NewPopulation = []
        for i in range(ParentsList.shape[1]):
            parent_1, parent_2 = ParentsList[0][i], ParentsList[1][i]
            offspring = self.CrossOver(parent_1, parent_2, WeightLimit, Packs)
            NewPopulation.append(offspring)
        return NewPopulation

    
    def Mutation(self,offspring, WeightLimit, PacksList, Packs):   # to mutate offsprings depending on the given probability
        for m in range(int(len(offspring)*self.MutationProbability)):
            a = np.random.randint(0,len(PacksList))
            b = np.random.randint(0,len(offspring))
            offspring.insert(b, PacksList[a])
            
        offspring = self.CheckWeight(offspring, WeightLimit, Packs)
        offspring = list(set(offspring))    # remove duplications
        return offspring


    def MutatePopualation(self,NewPopulation, WeightLimit, PacksList, Packs): # mutating all individuals using Mutation()  
        MutatedPopulation = []
        for offspring in NewPopulation:
            MutatedPopulation.append(self.Mutation(offspring, WeightLimit, PacksList, Packs))
        return MutatedPopulation



    def Solve(self,Packs,PacksList,Iterations):

        population = self.InitialPopulation(PacksList, self.PopulationSize, self.WeightLimit, Packs)
        FitnessList = self.Fitness(population,Packs)
        ParentsList = self.RouletteSelection(population,FitnessList)
        NewPopulation = self.CrossoverPairs(ParentsList, self.WeightLimit, Packs)
        MutatedPopulation = self.MutatePopualation(NewPopulation, self.WeightLimit,PacksList, Packs)
        BestResult = [-1,-np.inf,np.array([])] # to assign the best value and solution for each iteration
        for i in range(Iterations):
            FitnessList = self.Fitness(MutatedPopulation,Packs)
            if FitnessList.max() > BestResult[1]:
                BestResult[1] = FitnessList.max()
                BestResult[2] = np.array(MutatedPopulation)[FitnessList.max() == FitnessList]
            ParentsList = self.RouletteSelection(population,FitnessList)
            NewPopulation = self.CrossoverPairs(ParentsList, self.WeightLimit, Packs)
            MutatedPopulation = self.MutatePopualation(NewPopulation, self.WeightLimit,PacksList, Packs)
            return [BestResult[1] , BestResult[2][0]]




def Dynamic(WeightLimit, weight, value, length):
   K = [[0 for x in range(WeightLimit + 1)] for x in range(length + 1)]
   for i in range(length + 1):
      for w in range(WeightLimit + 1):
         if i == 0 or w == 0:
            K[i][w] = 0
         elif weight[i-1] <= w:
            K[i][w] = max(value[i-1] + K[i-1][w-weight[i-1]], K[i-1][w])
         else:
            K[i][w] = K[i-1][w]
   return K[length][WeightLimit]





def home(request):
    form = packValues(request.POST or None)
    if form.is_valid():
        form.save

    return render(request,'home.html',{'form1': form , 'form2' : limit})



def result(request):
    Packs={}
    PacksList=[]
    p = pack.objects.all().values()

    for i in range(len(p)+1):
        Packs[i+1] =[ p[i]['weight'] , p[i]['value'] ]
        PacksList.append(i+1)

    form = limit(request.POST)
    if form.is_valid():
        WeightLimit = request.POST['WeightLimit']
        Iterations = request.POST['Iterations']
        MuatationProb = request.POST['MutationProbabilty']

        ### solving using genetic algorithm
        genetic = GA(WeightLimit,MuatationProb)
        PacksList = np.array(PacksList)
        res = genetic.Solve(Packs,PacksList,Iterations)
        MaxValue = res[0] 
        SelectedPacks= res[1]

        ### solving using Dynamic programming
        length = len(Packs)
        weightList = [p['weight'] for i in Packs]
        ValueList = [p['value'] for i in Packs]
        DMaxValue = Dynamic(WeightLimit,weightList,ValueList,length)

    return render(request,'result.html',{'Value' : MaxValue , 'Packs' : SelectedPacks , 'Dvalue' : DMaxValue})