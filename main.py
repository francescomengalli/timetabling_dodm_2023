# works with Python 3.9.7
# importing the necessary libraries

import gurobipy as gp
import numpy as np
import pandas as pd
import argparse
import os

# defining the enrol matrix: it returns a 0-1 matrix where:
# 1: the student s_i is enrolled for exam e_j
# 0: otherwise

def do_enrol_matrix(instance):
    enrol_matrix = pd.read_table("instances/" + instance + ".stu", sep=" ", names=["STUDENT", "EXAM"])
    enrol_matrix = np.array(pd.crosstab(enrol_matrix['STUDENT'], enrol_matrix['EXAM']).astype(bool).astype(int))    
    n_students = np.shape(enrol_matrix)[0]
    n_exams = np.shape(enrol_matrix)[1]
    return enrol_matrix, n_students, n_exams


# defining the conflict matrix (simmetric)
# the entries of this matrix are integers that count the number of students enrolled for both exam e_i and e_j

def do_conflict_matrix(enrol_matrix):
    n_exams = np.shape(enrol_matrix)[1]
    conflict_matrix = np.zeros((n_exams, n_exams))
    for exam_1 in range(n_exams):
        for exam_2 in range(exam_1 + 1, n_exams):
            conflict_matrix[exam_1,exam_2] = np.sum([stud[exam_1]*stud[exam_2] for stud in enrol_matrix])
    return conflict_matrix

# creating the model

def create_model(model_name='ExamTimetabling', time_limit=1000, pre_solve=-1, mip_gap=1e-4, threads=4):
    model = gp.Model(model_name) # setting the name of the model
    model.setParam('TimeLimit', time_limit) # setting the time limit for the model (this is the default value)
    model.setParam("Presolve", pre_solve) # setting the level of pre-solving of the model (if "-1" -> default value)
    model.setParam("MIPGap", mip_gap) # optimality gap for the mix integer programming problem (MIP)
    model.setParam("Threads", threads) # number of treads
    model.setParam("NodefileStart", 0.5) # usage of memory for saving
    return model


# defining the variables for the model

def do_variables(n_exams, n_timeslots, model):
    x = {} # dictionary for variables x (assignment of exams to timeslots)
    y = {} # dictionary for variables y (indicator of overlap between pairs of exams in the same timeslot)
    z = {} # dictionary for variables z (indicator of timeslot utilization)

    # loop through timeslots
    for timeslot in range(n_timeslots):
        z[timeslot] = model.addVar(vtype=gp.GRB.BINARY, name=f'z[{timeslot}]') # adding the variable z[timeslot] to the model as binary (0 or 1)

        # loop through exams
        for exam in range(n_exams):
            x[exam, timeslot] = model.addVar(vtype=gp.GRB.BINARY, name=f'x[{exam},{timeslot}]') # adding the variable x[exam, timeslot] to the model as binary (0 or 1)

            # loop through subsequent pairs of exams
            for exam_2 in range(exam+1, n_exams):
                y[exam, exam_2, timeslot] = model.addVar(vtype=gp.GRB.BINARY, name=f'y[{exam},{exam_2},{timeslot}]') # adding the variable y[exam, exam_2, timeslot] to the model as binary (0 or 1)

    return x, y, z

def do_obj_function(measure, n_exams, n_students, n_timeslots, conflict_matrix, x):
    # initializing the objective function value
    obj_function = 0

    # Check the selected measure for the objective function
    if measure == "penalty":
        # iterate through pairs of exams
        for exam_1 in range(n_exams):
            for exam_2 in range(exam_1 + 1, n_exams):
                # check if there is a conflict between the exams
                if conflict_matrix[exam_1, exam_2] > 0:
                    # iterate through timeslots for both exams
                    for timeslot_1 in range(n_timeslots):
                        for timeslot_2 in range(max(0, timeslot_1 - 5), min(timeslot_1 + 6, n_timeslots)):
                            # update the objective function based on penalty measure
                            obj_function += 2**(5 - abs(timeslot_1 - timeslot_2)) * conflict_matrix[exam_1, exam_2] / n_students * x[exam_1, timeslot_1] * x[exam_2, timeslot_2]

    elif measure == "b2b":
        # iterate through timeslots and pairs of exams
        for timeslot in range(n_timeslots - 1):
            for exam_1 in range(n_exams):
                for exam_2 in range(exam_1 + 1, n_exams):
                    # update the objective function based on back-to-back measure
                    obj_function += conflict_matrix[exam_1, exam_2] * (x[exam_2, timeslot + 1] * x[exam_1, timeslot] + x[exam_2, timeslot] * x[exam_1, timeslot + 1])

    else:
        # print an error message if the selected measure is not implemented
        print("Warning: this measure is not available in this project. Type 'penalty' or 'b2b'.")

    return obj_function


def add_constraints(model_type, n_exams, n_timeslots, conflict_matrix, model):
    # Check the type of model to determine which constraints to add
    if model_type == "base":
        # CONSTRAINT 1: Each exam is scheduled exactly once
        for exam in range(n_exams):
            model.addConstr(gp.quicksum(x[exam, timeslot] for timeslot in range(n_timeslots)) == 1)

        # CONSTRAINT 2: Conflicting exams cannot be scheduled in the same time-slot
        for exam_1 in range(n_exams):
            for exam_2 in range(exam_1+1, n_exams):
                if conflict_matrix[exam_1, exam_2] > 0:
                    for timeslot in range(n_timeslots):
                        model.addConstr(x[exam_1, timeslot] + x[exam_2, timeslot] <= 1)
    
    elif model_type == "advanced":
        # CONSTRAINT 1: Each exam is scheduled exactly once
        for exam in range(n_exams):
            model.addConstr(gp.quicksum(x[exam, timeslot] for timeslot in range(n_timeslots)) == 1)

        # CONSTRAINT 2: At most 3 consecutive time slots can have conflicting exams
        for timeslot in range(n_timeslots):
            model.addConstr(gp.quicksum(x[exam_1, timeslot] * x[exam_2, timeslot]
                                        for exam_1 in range(n_exams)
                                        for exam_2 in range(exam_1+1, n_exams)
                                        if conflict_matrix[exam_1, exam_2] > 0) <= 1000 * z[timeslot])
            model.addConstr(z[timeslot] <= gp.quicksum(x[exam_1, timeslot] * x[exam_2, timeslot]
                                                        for exam_1 in range(n_exams)
                                                        for exam_2 in range(exam_1+1, n_exams)
                                                        if conflict_matrix[exam_1, exam_2] > 0))
        
        for timeslot in range(n_timeslots-3):    
            model.addConstr(5 - gp.quicksum(z[timeslot+i] for i in range(3)) >= 3 * z[timeslot+3])
                
        for timeslot in range(1, n_timeslots-3):    
            model.addConstr(5 - gp.quicksum(z[timeslot+i] for i in range(3)) >= 3 * z[timeslot-1])

        # CONSTRAINT 3: If two consecutive time slots contain conflicting exams, then no conflicting exam can be scheduled in the next 3 time slots
        for timeslot in range(n_timeslots-4):
            for exam_1 in range(n_exams):
                for exam_2 in range(exam_1+1, n_exams):
                    if conflict_matrix[exam_1, exam_2] > 0:
                        model.addConstr(y[exam_1, exam_2, timeslot] == x[exam_1, timeslot] * x[exam_2, timeslot+1] + x[exam_2, timeslot] * x[exam_1, timeslot+1])
                        model.addConstr(gp.quicksum(x[exam, t] * y[exam_1, exam_2, timeslot]
                                                    for t in range(timeslot+2, timeslot+5)
                                                    for exam in range(n_exams)
                                                    if exam != exam_1 and exam != exam_2
                                                    and (conflict_matrix[exam, exam_1] > 0 or conflict_matrix[exam, exam_2] > 0)) == 0)

        # CONSTRAINT 4: Change the constraints that impose that no conflicting exams can be scheduled in the same time slot. 
        #               Instead, impose that at most 3 conflicting pairs can be scheduled in the same time slot.
        for timeslot in range(n_timeslots):
            model.addConstr(gp.quicksum(x[exam_1, timeslot] * x[exam_2, timeslot]
                                        for exam_1 in range(n_exams)
                                        for exam_2 in range(exam_1+1, n_exams)
                                        if conflict_matrix[exam_1, exam_2] > 0) <= 3)

    else:
        # Print an error message if the selected model type is invalid
        print("Warning: this type of model is not valid. Type 'base' or 'advanced'.")
                            
    return 0


def solution(folder, instance, model, x, n_exams, n_timeslots):
    # verify if the directory exists, otherwise create it
    utput_directory = "solutions/base_penalty/"
    os.makedirs(output_directory, exist_ok=True)
    # open the solution folder for saving the solution
    file = open(output_directory + instance + ".sol", "w")  # 'w' for selecting "write mode"
    
    # printing the solution
    if model.status == gp.GRB.INFEASIBLE:
        # if the model is infeasible, write "INFEASIBLE" to the solution file
        file.write("INFEASIBLE")
    elif model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        # if the model is optimal or reached the time limit, write the exam assignments to the solution file
        for exam in range(n_exams):
            for timeslot in range(n_timeslots):
                if x[exam, timeslot].x > 0.5:
                    # write the assignment in the format "exam timeslot" to the solution file
                    file.write(f'{str(exam+1).zfill(4)} {timeslot+1}\n')                

    # close the solution file
    file.close()
    
    return 0


#####################################################################################################################################################

# Main section of the script: Handles command-line arguments, sets up the optimization model, and performs optimization


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(
    	description="This utility assesses the viability of a given solution for a specific instance. Please supply the solution file along with the necessary input data files (.stu and .slo).",
    	epilog="""-------------------""")

    # Required positional arguments (given by the user)
    parser.add_argument("instance", type=str,
                        help="[string] required argument - name of the instance ('test' or 'instanceXX').")
    parser.add_argument("measure", type=str,
                        help="[string] required argument - name of the objective function ('penalty' or 'b2b').")
    parser.add_argument("model_type", type=str,
                        help="[string] required argument - type of the model ('base' or 'advanced').")
    parser.add_argument("time_limit", type=int,
                        help="[int] required argument - time limit.")

    args = parser.parse_args()

    # Extract arguments from the parser
    instance = args.instance
    measure = args.measure
    model_type = args.model_type
    model_time_limit = args.time_limit

    # Read input data
    enrol_matrix, n_students, n_exams = do_enrol_matrix(instance)
    conflict_matrix = do_conflict_matrix(enrol_matrix)
    n_timeslots = int(list(pd.read_table("instances/" + instance + ".slo").columns)[0])

    # Create and configure the optimization model
    model = create_model(model_name='Exam_Time_tabling', time_limit=model_time_limit, pre_solve=-1, mip_gap=1e-4, threads=4)
    
    # Create decision variables
    x, y, z = do_variables(n_exams, n_timeslots, model)
    
    # Set the objective function
    obj_function = do_obj_function(measure, n_exams, n_students, n_timeslots, conflict_matrix, x)
    model.setObjective(obj_function, gp.GRB.MINIMIZE)
    
    # Add constraints to the model
    add_constraints(model_type, n_exams, n_timeslots, conflict_matrix, model)
    
    # Optimize the model
    model.optimize()
    
    # Output the solution
    solution(model_type + "_" + measure, instance, model, x, n_exams, n_timeslots)


#####################################################################################################################################################
