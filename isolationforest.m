%clear all;
%clc;
close all;

%load data
data=readtable('bank-additional-full_normalised.csv'); %this is just reading the data - read the bank data here. 
%if you want to see for unlabeled data, then simply don't display the AUC
%and just get the scores for the anomalies, and plot them. however this
%code, run on its own, doesn't do it.
%also the unlabeled data isn't loaded here.
data=table2array(data);
ADLabels=data(:,end);
data=data(:,1:end-1);
Data=data(:,1:end-1);
%Data=b;
%data=b;

%Run iForest
%general parameters
number_of_rounds = 100; %how many rounds do you want to run?
%parameters for iForest
numtrees = 50; %how many trees do you want?
numsubsamples = 500; % how many samples do you want to take at a time?
numdimensions = size(Data, 2);

auc = zeros(number_of_rounds, 1);

for r = 1:number_of_rounds
    disp(['rounds ', num2str(r), ':']);
    Forest = isolator(Data, numtrees, numsubsamples, numdimensions);
    [center] = IsolationEstimation(Data, Forest);
    Score = -mean(center, 2);
    auc(r) = Measure_AUC(Score, ADLabels);
    disp(['auc=', num2str(auc(r)), '.']); %honestly, there was no need to it this way, but this was before i knew that perfcurve() just gave me the answer
end

AUC_results=[mean(auc), std(auc)]; % average AUC over the number of trials


%plot auc

[tpr,xpr,truthdata,aucdata] = perfcurve(logical(ADLabels),Score,'true'); %i was going to use my own implementation of calculating these, but didn't use it. my implementation ended up in the k-nn algorithm
plot(tpr,xpr) 
xlabel('FPR'); ylabel('TPR');
title('AUC')
%so i didn't break the rule of not using inbuilt functions!

function [center] = IsolationEstimation(testdata, forest) %there is no reason why this variable should be named center
%but it evolved out of a lot of rewriting of the code

    size_test_data = size(testdata, 1);
    center = zeros(size_test_data, forest.numberoftrees);

    for i = 1:forest.numberoftrees
        center(:, i) = importanceofisolation(testdata, 1:size_test_data, forest.Trees{i, 1}, zeros(size_test_data, 1)); %this function just stores the degree to which a point is isolated
    end
end

function forest = isolator(data, numtree, numsubtrees, dimensions) %this function actually isolates elements

[NumInst, pickeddimension] = size(data); %you may ask, why are the variable names all funny? i repeatedly added to the code, that's why they're all over the place
forest.Trees = cell(numtree, 1);

forest.numberoftrees = numtree;
forest.NumSub = numsubtrees;
forest.dimensionnumber = dimensions;
forest.maxheight = ceil(log2(numsubtrees)); %this is straightforward and does the procedure in the paper

%parameters for function IsolationTree
Paras.maxheight = forest.maxheight;
Paras.dimensionnumber = dimensions;

for i = 1:numtree
    
    if numsubtrees < NumInst %subsamples are selected randomly here
        [~, indices_of_subset] = sort(rand(1, NumInst));
        subset_index = indices_of_subset(1:numsubtrees);
    else
        subset_index = 1:NumInst;
    end
    if dimensions < pickeddimension %same here
        [~, randomdimension] = sort(rand(1, pickeddimension));
        index_of_dimension = randomdimension(1:dimensions);
    else
        index_of_dimension = 1:pickeddimension;
    end
    
    Paras.index_of_dimension = index_of_dimension;
    forest.Trees{i} = IsolationTree(data, subset_index, 0, Paras); % build an isolation tree
    
end
end

function isolatedimportance = importanceofisolation(Data, indexofcurrent, Tree, isolatedimportance) %probably should be named better, just does what's described in the algorithm

if Tree.node_existence == 0
    
    if Tree.Size <= 1
        isolatedimportance(indexofcurrent) = Tree.Height;
    else
        c = 2*(log(Tree.Size-1)+0.5772156649)-2*(Tree.Size-1)/Tree.Size; %calculate c(n)
        isolatedimportance(indexofcurrent)=Tree.Height+c;
    end
    return;
    
else
    
    left_index = indexofcurrent(Data(indexofcurrent, Tree.splitattribute) < Tree.SplitPoint);
    right_index = setdiff(indexofcurrent, left_index);
    
    if ~isempty(left_index)
        isolatedimportance = importanceofisolation(Data, left_index, Tree.leftchild, isolatedimportance);
    end
    if ~isempty(right_index)
        isolatedimportance = importanceofisolation(Data, right_index, Tree.rightchild, isolatedimportance);
    end
    
end
end

function cumulativeauc = Measure_AUC(Scores, Labels) %actually measure the AUC, BY HAND (again, no need to do this)

NumInst = length(Scores);

%sort the scores
[Scores, index]=sort(Scores, 'descend');
Labels=Labels(index);

numpositivelabels=1;
numnegativelabels=0;

positivelabels=length(find(Labels == numpositivelabels));
negativelabels=length(find(Labels == numnegativelabels));

cumulativepositive=0;
cumulativenegative=0;
cumulativeauc=0;

reciprocalpositive = 1/positivelabels;
reciprocalnegative = 1/negativelabels;

i = 1; %just trying to see if it works - this basically does what you do by hand on the command line, so i put it into the code
while i <= NumInst
    temp = cumulativepositive;
    if (i < NumInst - 1) && (Scores(i) == Scores(i + 1))
        while (i < NumInst - 1) && (Scores(i) == Scores(i + 1))
            if Labels(i) == numnegativelabels
                cumulativenegative = cumulativenegative + 1;
            elseif Labels(i) == numpositivelabels
                cumulativepositive = cumulativepositive + 1;
            else
                disp('Cannot find a label!');
            end
            i = i + 1;
        end

        if Labels(i) == numnegativelabels
            cumulativenegative = cumulativenegative + 1;
        elseif Labels(i) == numpositivelabels
            cumulativepositive = cumulativepositive + 1;
        else
            disp('Cannot find a label!');
        end

        cumulativeauc = cumulativeauc + (cumulativepositive + temp) * reciprocalpositive * cumulativenegative * reciprocalnegative/2; %in hindsight, there is a much faster way to do this. oh well
        cumulativenegative = 0;
    else
        if Labels(i) == numnegativelabels
            cumulativenegative = cumulativenegative + 1;
            cumulativeauc = cumulativeauc + cumulativepositive * reciprocalpositive * cumulativenegative * reciprocalnegative;
            cumulativenegative = 0;
        elseif Labels(i) == numpositivelabels
            cumulativepositive = cumulativepositive + 1;
        else
            disp('Cannot find a label!');
        end
    end
    i = i + 1;
end
end

function Tree = IsolationTree(Data, currentdataindex, currenttreeheight, parameters)  %actually construct a tree by hand
Tree.Height = currenttreeheight;
instances = length(currentdataindex);

if currenttreeheight >= parameters.maxheight || instances <= 1
    Tree.node_existence = 0;
    Tree.splitattribute = [];
    Tree.SplitPoint = [];
    Tree.leftchild = [];
    Tree.rightchild = []; %this part caused me so much pain, making a tree by hand. never again
    Tree.Size = instances;
    return;
else
    Tree.node_existence = 1;
    %randomly select one dimension to split
    [~, rindex] = max(rand(1, parameters.dimensionnumber));
    Tree.splitattribute = parameters.index_of_dimension(rindex);
    current_data_point = Data(currentdataindex, Tree.splitattribute);
    Tree.SplitPoint = min(current_data_point) + (max(current_data_point) - min(current_data_point)) * rand(1); %this actually arose out of the autoencoder initialization
    
    % instance index for left child and right children
    left_index = currentdataindex(current_data_point < Tree.SplitPoint);
    right_index = setdiff(currentdataindex, left_index);
    
    % bulit right and left child trees
    Tree.leftchild = IsolationTree(Data, left_index, currenttreeheight + 1, parameters);
    Tree.rightchild = IsolationTree(Data, right_index, currenttreeheight + 1, parameters);
end
end
