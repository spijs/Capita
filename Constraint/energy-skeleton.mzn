% data
%%%%%%
int: nbMachines;
int: nbTasks;
int: nbRes;
set of int: Machines = 1..nbMachines;
set of int: Tasks = 1..nbTasks;
set of int: Res = 1..nbRes; % Resources

% Machine resources
array[Machines,Res] of int: m_res;

% Tasks
array[Tasks] of int: j_earl; % earliest start
array[Tasks] of int: j_late; % latest end
array[Tasks] of int: j_dur; % duration
array[Tasks] of float: j_power; % power use per time unit
array[Tasks,Res] of int: j_res;

% Energy
int: time_step;
float: q = int2float(time_step);
set of int: Times = 1..(round(24.0*60.0/q));
array[Times] of float: price;


% decis. variables
%%%%%%%%%%%%%%%%%%
% x_jmt: task j starts at time t on machine m
array[Tasks,Machines,Times] of var 0..1: x;

% Task constraints
%%%%%%%%%%%%%%%%%%
% each task is scheduled only once
array[Tasks] of var 0..(nbMachines*max(Times)): nbOfSchedules;
constraint forall(i in 1..nbTasks)(
                  nbOfSchedules[i] = sum (j in 1..nbMachines, t in Times)(x[i,j,t])
                  /\ nbOfSchedules[i] == 1  
);
% no task can be scheduled before its earliest start
constraint forall(i in Tasks, j in Machines, t in Times where t < j_earl[i])(x[i,j,t] == 0);


% no task can be scheduled after its latest end
constraint forall(i in Tasks, j in Machines, t in Times where t + j_dur[i] > j_late[i])(x[i,j,t] == 0);

% respect the max load on each machine
array[Machines,Times,Res] of var float: totalLoads;

constraint forall(m in Machines, r in Res, t in Times)(
    totalLoads[m,t,r] = sum (task in Tasks, t2 in max(1,t-j_dur[task]+1)..t)  (x[task,m,t2]*j_res[task,r])
    /\ totalLoads[m,t,r] <= m_res[m,r]);
% Objective function
%%%%%%%%%%%%%%%%%%%%

var float: c_task = sum (t in Times, m in Machines, task in Tasks) 
(x[task,m,t] * sum(t2 in t..min(max(Times),(t+j_dur[task]-1)))(j_power[task]*price[t2])); 
% var float : c_task = 0.0;
solve minimize c_task;

output  [ "Machine=\(m),Start=\(t),Task=\(j)\n" | m in Machines, t in Times, j in Tasks where fix(x[j,m,t])==1]++
        [ "Cost=\(c_task)" ];
