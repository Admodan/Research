import pydicom
import glob 
import numpy as np
import re
import pandas as pd
import bisect

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks, shape_index
from scipy import ndimage as ndi
from skimage.feature import CENSURE

# General Purpose Beam Information

def get_total_beam_MU(plan_dataset,beam_num):
    """ Returns the total beam MU as a float pulled from the plan file. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match. 
        Returns:
            totalMU      : Total MU planned for this beam 
    """
    totalMU = plan_dataset.FractionGroupSequence[0].ReferencedBeamSequence[beam_num].BeamMeterset
    return totalMU

def get_max_gantry_speed(plan_dataset):
    """ Uses hardcoded values via Lukas to return the maximum gantry speed for the unit found in the plan. 
        May need to be modified with new units being installed
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            
        Returns:
            maxgantryspeed      : Maximum gantry speed 
    """
    unitnum = get_unit_number(plan_dataset)
    maxgantryspeed = 0
    if unitnum == None:
        print("Non-integer unit number, MaxGantrySpeed assumed 6deg/sec")
        maxgantryspeed=6.0 #as above 
        return maxgantryspeed
    if unitnum == 10:
        maxgantryspeed=4.8 #via Lukas, assumed deg/sec
        return maxgantryspeed
    else:
        maxgantryspeed=6.0 #as above 
        return maxgantryspeed

    
def get_dose_rate_set(plan_dataset,beam_num):
    """ Returns the maximum dose rate for this plan. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            drset        : Dose rate set for this plan
            
        Raises : 
            TypeError    : If the indexing is non-traditional, this will raise and invalidate the plan. 
    """
    drset = plan_dataset.BeamSequence[beam_num].ControlPointSequence[0].DoseRateSet
    try:
        drset = int(drset)
        return drset
    except TypeError:
        print("Non-integer returned for Dose Rate")
        return None    

def get_cp_count(plan_dataset,beam_num):
    """ Returns the number of control points for this beam. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            cpcount        : Integer value, number of control points 
    """
    cpcount = plan_dataset.BeamSequence[beam_num].NumberOfControlPoints
    return cpcount

def calc_SDT(dangle,dmu,maxgantryspeed,maxdoserate):
    """ Calculates the Segment Delivery Times (time between control points using two methods, then finds maximum possible time. 
    
        Parameters:
            dangle         : Delta Angle matrix (change in gantry angles), via get_delta_gantry_angle_matrix(...) 
            dmu            : Delta MU matrix (change in delivered MU)    , via get_delta_MU_matrix(...)
            maxgantryspeed : Maximum gantry speed for this arc           , via get_max_gantry_speed(...)
            maxdoserate    : Dose rate set for this arc                  , via get_dose_rate_set(...)
            
         Returns: 
            maxTime        : A matrix of the segment delivery times, element-wise maximum possible time for each entry.  
    """
    #segment is portion between control points eh. dangle and dmu. 
    maxdoserate = maxdoserate/60 #convert to seconds      
    RotTimeViaMaxSpeed   = np.absolute(dangle / maxgantryspeed)
    DelivTimeViaDoseRate = np.absolute(dmu   /  maxdoserate) 
    maxTime = np.around(np.maximum(RotTimeViaMaxSpeed,DelivTimeViaDoseRate),decimals=2)
    return maxTime


def get_lumped_leaf_positions(plan_dataset,beam_num,cpnum):
    """ Returns the MLC leaf positions directly pulled from the plan dataset. 
        Function can be used iteratively to information of entire arc.
        
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            cpnum        : The control point currently being examined 
            
        Returns:
            lumpbank     : A numpy matrix containing all of the leaf positions for a given beam, given control point. 
                           Contains 2*N values, where N is the Number of Leaf/Jaw Pairs(element) subscript order 101, 102, … 1N, 201, 202, … 2N.
    """
    len_sequence = len(plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence)
    if len_sequence == 3:
        lumpbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions
        return lumpbank
    elif len_sequence == 1:
        lumpbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[0].LeafJawPositions
        return lumpbank
    else:
        return None
        

def get_unit_number(plan_dataset,beamnum=0):
    """ Utility function, number is grabbed as formality but units 3 & 5 are often used interchangably. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of any beam in plan, will be the same unit for any
        Returns:
            cpcount      : Integer value, number of control points 
    """
    namer = plan_dataset.BeamSequence[beamnum].TreatmentMachineName
    unitnum = namer[4] #hardcoding as field is standard naming:'unit5ser2899'
    try:
        unitnum = int(unitnum)
        return unitnum
    except TypeError:
        print("Non-integer returned for Unit Number")
        return None
        
        
# MLC Speed Score functions 

def calc_dr_and_gs_at_cp_matrix(plan_dataset,beam_num):
    """ Calculates the Dose Rate and Gantry Speed matrices (used side by side later, so calculated together) 
             
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            doserate     : Matrix where each entry is the dose rate at the indexed control point, rounded to 2 sigfig
            gantryspeed  : Matrix where each entry is the gantry speed at the indexed control point, rounded to 2 sigfig
    """
    
    totalbeamMU = get_total_beam_MU(plan_dataset,beam_num)
    
    dangle = get_delta_gantry_angle_matrix(plan_dataset,beam_num)
    dmu = get_delta_MU_matrix(plan_dataset,beam_num)
    maxgs = get_max_gantry_speed(plan_dataset)  #degrees per second
    maxdr = get_dose_rate_set(plan_dataset,beam_num) # is in MU/min, /60 converts 
    segdeltime = calc_SDT(dangle,dmu,maxgs,maxdr)

    doserate = dmu/segdeltime *60
    gantryspeed = np.absolute(dangle/segdeltime)

    return np.around(doserate,decimals=2),np.around(gantryspeed,decimals=2)

def get_gantry_angle_matrix(plan_dataset,beam_num):
    """ Gets the gantry angle for all control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            gangle       : Matrix of gantry angles where each entry is the gantry angle at indexed control point
    """
    gangle = [] # to build matrix for angles for all control points
    for i in range(0,get_cp_count(plan_dataset,beam_num)):
        gangle.append(plan_dataset.BeamSequence[beam_num].ControlPointSequence[i].GantryAngle)
    gangle = np.array(gangle) 

    return gangle

def get_delta_gantry_angle_matrix(plan_dataset,beam_num):
    """ Gets the change in gantry angle between control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            dangle       : Matrix of gantry angles where each entry is the gantry angle between each indexed control point. 
                           for N control points, this has dimension N-1.  
    """
    gangle = get_gantry_angle_matrix(plan_dataset,beam_num)
    dangle = np.diff(gangle)
    dangle = np.where(dangle>180,360-dangle,dangle)
    dangle = np.absolute(dangle)
    return dangle

def get_MU_matrix(plan_dataset,beam_num):
    """ Gets the MU matrix as calculated from Lukas' code for Aperture score 
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            MUcorrected  : Matrix of Meterset weighting at each control point against total beam MU
    """
    muraw = []
    totalBeamMU = get_total_beam_MU(plan_dataset,beam_num)
    for i in range(0,get_cp_count(plan_dataset,beam_num)):
        muraw.append(plan_dataset.BeamSequence[beam_num].ControlPointSequence[i].CumulativeMetersetWeight)
    mu = muraw #swap for above to revert and test as-if non-culmulative meterset 
    MUcorrected = np.diff(np.array(mu)) * totalBeamMU 
    #(CulMetersetWeight-CulMetersetWeightPrev) #multiplies weight against dose rate
    #need to subtract previous CulMeterWeight as is culmulative value from start of treatment. 

    return MUcorrected

def get_delta_MU_matrix(plan_dataset,beam_num):
    """ Gets the change in MU for all control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            dmu       : Matrix of gantry angles where each entry is the gantry angle between indexed control points
                        for N control points, this has dimension N-1.  
    """
    mumatrix = get_MU_matrix(plan_dataset,beam_num)
    #dmu = np.diff(mumatrix)
    dmu = mumatrix
    return dmu 

def get_number_of_beams(plan_dataset):
    """ Utility function: The number of beams in this plan file  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)

        Returns:
            beam_num     : Integer value, number of beams in plan.  
    """
    beam_num = len(plan_dataset.BeamSequence)
    return beam_num


def calc_gantry_speed_complexity_score(plan_dataset,beam_num, type="vel"):
    """ Calculates two types of complexity scores based on the gantry modulation: velocity or acceleration
        Each is a weighted sum of the values, still work in progress to find best results. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            type         : String, either 'vel' or 'acc' depending on which score is desired. 
        Returns:
            gsscore       : Normalized value for the gantry speed modulation score 
    """
    controlpointtime = calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num)
    if np.isnan(controlpointtime).sum() > 0: #if any values are NaN, use other matrix. 
        print("NaN encountered in get_all_mlc_speeds_matrix")
        controlpointtime = calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num)
    
    controlpointveltime = controlpointtime
    controlpointacctime = np.diff(controlpointveltime)
    gs_dpos_matrix = get_delta_gantry_angle_matrix(plan_dataset,beam_num)
    
    gs_ddpos_matrix = np.diff(gs_dpos_matrix)
    
    if type == "vel":
        gs_matrix = np.true_divide(gs_dpos_matrix,controlpointveltime)
    if type == "acc":
        gs_matrix = np.true_divide(gs_ddpos_matrix,controlpointacctime)
    gs_matrix[np.isnan(gs_matrix)] = 0
    
    gs_score = np.absolute(gs_matrix).sum()
    
    MU_mat = normalize_peak_to_peak(get_MU_matrix(plan_dataset,beam_num)) #normalized to set ~0 as 0
    num_of_zero_MU_control_points= np.count_nonzero(MU_mat==0)
    
    
    max_gantry = get_max_gantry_speed(plan_dataset)
    cp_count = get_cp_count(plan_dataset,beam_num)-num_of_zero_MU_control_points
    normalization_val = cp_count*max_gantry
    
    return gs_score/normalization_val

def calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num):
    """ Gets the change in MU for all control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            dmu       : Matrix of gantry angles where each entry is the gantry angle between indexed control points
                        for N control points, this has dimension N-1.  
    """
    dangle = get_delta_gantry_angle_matrix(plan_dataset,beam_num)
    dr, gs = calc_dr_and_gs_at_cp_matrix(plan_dataset,beam_num)
    treattime_gs = dangle/gs 
    return treattime_gs

def calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num):    
    totalbeamMU = get_total_beam_MU(plan_dataset,beam_num)
    dmu = get_delta_MU_matrix(plan_dataset,beam_num)
    deliveredmu = dmu#* totalbeamMU
    dr, gs = calc_dr_and_gs_at_cp_matrix(plan_dataset,beam_num)
    treatmentminutes = deliveredmu/dr
    treatmentsecondsmatrix = np.nan_to_num(treatmentminutes *60)
    return treatmentsecondsmatrix

def calc_total_time_from_treatment_times(times_matrix):
    res = np.sum(times_matrix)
    return res



def get_all_mlc_speeds_matrix(plan_dataset,beam_num):
    controlpointtime = calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num)
    if np.isnan(controlpointtime).sum() > 0: #if any values are NaN, use other matrix. 
        print("NaN encountered in get_all_mlc_speeds_matrix")
        controlpointtime = calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num)
    
    leaf_pos_list = []
    for i in range(0,get_cp_count(plan_dataset,beam_num)):
        leaf_pos_list.append(get_lumped_leaf_positions(plan_dataset,beam_num,i))
        
    leaf_pos_arr = np.array(leaf_pos_list)

    
    
    #here, split into 60-60. weight outer 16's by factor of 2. recombine.  
    
    
    
    
    leaf_diff_arr = np.absolute(np.diff(leaf_pos_arr,axis=0))
    single_leaf = leaf_diff_arr.T
    
    leaf_speeds = np.absolute(np.true_divide(leaf_diff_arr,controlpointtime[:,None])) # divide each control point leaf positions vector by time
    return leaf_speeds

def get_all_mlc_acc_matrix(plan_dataset,beam_num):
    controlpointtime = calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num)
    if np.isnan(controlpointtime).sum() > 0: #if any values are NaN, use other matrix. 
        print("NaN encountered in get_all_mlc_speeds_matrix")
    controlpointacctime = controlpointtime[:-1]
    controlpointtime = calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num)
    
    leaf_speed = get_all_mlc_speeds_matrix(plan_dataset,beam_num)

    leaf_speed_diff_arr =np.absolute(np.diff(leaf_speed,axis=0)) #np.pad(np.absolute(np.diff(leaf_speed,axis=0)), (1, 0), mode='constant')

    leaf_acc = np.absolute(np.true_divide(leaf_speed_diff_arr,controlpointacctime[:,None]))
    
    return(leaf_acc)

 
def calc_MLC_beam_acc_complexity(plan_dataset,beam_num): 
    mlc_acc = get_all_mlc_acc_matrix(plan_dataset,beam_num)

    splitter = np.split(mlc_acc,2,axis=1) #yuh

    mlc_acc_left = splitter[0] #split left bank, right bank
    mlc_acc_right = splitter[1]


    standard_dev = np.std(mlc_acc)

    
    control_point_count = get_cp_count(plan_dataset,beam_num) -1  # due to np.diff 
    control_point_factor = 1/control_point_count  #multiplication is cheaper than division 
    
    count_over_left = (mlc_acc_left > standard_dev).sum()
    count_over_right = (mlc_acc_right > standard_dev).sum()

    adjusted_count_left = (count_over_left*control_point_factor)
    adjusted_count_right = (count_over_right*control_point_factor)
    
    total_adjusted_count = (adjusted_count_left+adjusted_count_right) 
    
    return total_adjusted_count,standard_dev    
    
def normalize_peak_to_peak(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d
    
    
def calc_MLC_beam_complexity(plan_dataset,beam_num,f_val=1): 
    MU_mat = normalize_peak_to_peak(get_MU_matrix(plan_dataset,beam_num)) #normalized to set ~0 as 0

    
    #print(f"MU Shape:{MU_mat.shape}")
    mlc_speeds = get_all_mlc_speeds_matrix(plan_dataset,beam_num)
    #print(f"MLC Speed Shape: {mlc_speeds.shape}")
    # np.multiply(mlc_speeds,MU_mat[:,None])
    #splitter = np.split(mlc_speeds_adjusted,2,axis=1) #yuh
    
    #mlc_speeds_left = splitter[0] #split left bank, right bank
    #mlc_speeds_right = splitter[1]
    

    standard_dev = f_val*np.std(mlc_speeds)

    #modify count, remove for each point delivered MU is = 0.
    num_of_zero_MU_control_points= np.count_nonzero(MU_mat==0)
    
    control_point_count = get_cp_count(plan_dataset,beam_num)-num_of_zero_MU_control_points  # due to np.diff 
    control_point_factor = 1/control_point_count  #multiplication is cheaper than division 

    
    # now find all indices for velocities which exceed the std, multiply them by a normalized dMU
    # this accounts for relatively high dose at a point which coincides with high leaf velocity
    
    #where_exceeds_std = np.where(mlc_speeds > standard_dev)
    #print(mlc_speeds.shape)
    count_weighted = 0
    for i in range (0,mlc_speeds.shape[0]-1):
        #where_exceeds_list.append(np.where(mlc_speeds[i] > standard_dev))
        #build loop score, then can convert to matrix form later
        #modify by leaf size. 
        delivered_MU = MU_mat[i]
        where_exceeds = np.where(mlc_speeds[i] > standard_dev)
        big_exceeds = np.count_nonzero(( (where_exceeds[0] >= 0) & (where_exceeds[0] <= 14) ) | ( (where_exceeds[0] >= 46) & (where_exceeds[0] <= 74) ) | ((where_exceeds[0] >= 106) & (where_exceeds[0] <= 120)))
        small_exceeds = np.count_nonzero(( (where_exceeds[0] > 14) & (where_exceeds[0] < 46) ) | ( (where_exceeds[0] > 74) & (where_exceeds[0] < 106) ) )
        count_weighted += (big_exceeds+small_exceeds)

            
        #print(delivered_MU)
    #this prints each CP. Then we also know the MU at each CP. So each of these mounts by relative dose modifier. 
    # and by index 0-16 BIG, etc, multiply nby size modifier

    
    count_over = (mlc_speeds > standard_dev).sum()
    #adjusted_count = (count_over*control_point_factor)
    #total_adjusted_count = adjusted_count
    
    total_count_weighted = count_weighted *control_point_factor

    return total_count_weighted,standard_dev

def calc_unified_mlc(plan_dataset,beam_num,f_range=[0.2,0.5,1,2]):
    res_list = []
    for f_val in f_range:
        res,trash = calc_MLC_beam_complexity(plan_dataset,beam_num,f_val)
        res_list.append(res)
    score = np.sum(np.array(res))
    return score
        
        
def calc_MLC_beam_complexity_classic(plan_dataset,beam_num): 
    MU_mat = normalize_peak_to_peak(get_MU_matrix(plan_dataset,beam_num))
    mlc_speeds = get_all_mlc_speeds_matrix(plan_dataset,beam_num)
    splitter = np.split(mlc_speeds,2,axis=1) #yuh

    mlc_speeds_left = splitter[0] #split left bank, right bank
    mlc_speeds_right = splitter[1]

    num_of_zero_MU_control_points= np.count_nonzero(MU_mat==0)
    standard_dev = np.std(mlc_speeds)

    
    control_point_count = get_cp_count(plan_dataset,beam_num) -num_of_zero_MU_control_points  # due to np.diff 
    control_point_factor = 1/control_point_count  #multiplication is cheaper than division 
    
    count_over_left = (mlc_speeds_left > standard_dev).sum()
    count_over_right = (mlc_speeds_right > standard_dev).sum()

    adjusted_count_left = (count_over_left*control_point_factor)
    adjusted_count_right = (count_over_right*control_point_factor)
    
    total_adjusted_count = (adjusted_count_left+adjusted_count_right) 
    
    return total_adjusted_count,standard_dev


# Aperture Complexity Score Functions 

def get_unit_number(plan_dataset,beamnum=0):
    namer = plan_dataset.BeamSequence[beamnum].TreatmentMachineName
    unitnum = namer[4] #hardcoding as field is standard naming:'unit5ser2899' # maybe use regex to optimize?
    try:
        unitnum = int(unitnum)
        return unitnum
    except TypeError:
        print("Non-integer returned for Unit Number")
        return None

def get_leaf_sizes(plan_dataset):
    unitnum = get_unit_number(plan_dataset)

    if (unitnum == 5 or unitnum == 3): 
        smallleaf, bigleaf, smallfromcenter = 2.5, 5, 16 #hardcoded from lukas
    else:
        smallleaf, bigleaf, smallfromcenter = 5, 10, 20  #hardcoded from lukas
    return smallleaf,bigleaf,smallfromcenter

def get_leaf_width(plan_dataset,currentleaf,totalleaves):
    if currentleaf < 1 or currentleaf > totalleaves :
        return None
    midpoint = totalleaves /2 # copied from lukas C#, hopefully does what its supposed to ?                   <--- * Validate 
    sleaf, bleaf, smallfromcenter = get_leaf_sizes(plan_dataset) #smallfrom center is how many are small, from center.
    if currentleaf > (midpoint-smallfromcenter) and currentleaf <= (midpoint +smallfromcenter):       
        return sleaf
    else:
        return bleaf # ! Think we can scratch this whole function. 
    
def calc_leaf_width_matrix(plan_dataset,beamnum,cpnum,total_leaves_per_side,weight=1):
    small,big,lcent = get_leaf_sizes(plan_dataset)
    mid = total_leaves_per_side
    matbank = get_leaf_positions_difference(plan_dataset,beamnum,cpnum,total_leaves_per_side)                     
    matbank[:(mid - lcent)] = matbank[:(mid - lcent)]*big #large outer leaves
    matbank[(mid + lcent):] = matbank[(mid + lcent):]*big #large outer leaves
    matbank[(mid - lcent):(mid + lcent)] = matbank[(mid - lcent):(mid + lcent)]*small*weight # small middle leaves
    return matbank    


def calc_leaf_weighted_area_matrix(plan_dataset,beamnum,cpnum,total_leaves_per_side,verbose=False):
    small,big,lcent = get_leaf_sizes(plan_dataset) # # of leaves from center where MLC transitions from using small to big leaf
    mid = total_leaves_per_side
    matbank = get_leaf_positions_difference(plan_dataset,beamnum,cpnum,total_leaves_per_side)  #raw area                   
    matbank[:(mid - lcent)] = matbank[:(mid - lcent)]*big #large outer leaves
    matbank[(mid + lcent):] = matbank[(mid + lcent):]*big #large outer leaves
    matbank[(mid - lcent):(mid + lcent)] = matbank[(mid - lcent):(mid + lcent)]*small # small middle leaves
    #matbank is now area at each leaf location
    #now add weighting factor based on 60 leaves
    # 60 areas, as 30 symmetric. Split into 3rds to start. Full weight to outer, .5 to middle, 0.25 to center? 
    #can try to optimize later if suggestive of success
    matbank[(mid - 20):(mid + 20)] = matbank[(mid - 20):(mid + 20)]*0.5
    matbank[(mid - 10):(mid + 10)] = matbank[(mid - 10):(mid + 10)]*0.5


    if verbose == True:
        print(matbank)
    return matbank   

def calc_edge_of_detector_metric(plan_dataset,beamnum,weight=1, ratio=[1,1], verbose=False):
    MU = get_total_beam_MU(plan_dataset,beamnum)
    cpcount = get_cp_count(plan_dataset,beamnum)
    score = 0
    for i in range(0,cpcount):
        muatcp = get_MU_at_CP(plan_dataset,beamnum,i)
        
        areaatcp = calc_control_point_aperture_area_jb(plan_dataset,beamnum,i,weight)
        res = np.round(muatcp * ( areaatcp),2)/MU
        score += res
    study_score = score/100 
    return study_score


def get_leaf_positions(plan_dataset,beam_num,cpnum):
    num_leaves = 60
    len_sequence = len(plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence)
    
#     print(f"{beam_num} of {len(plan_dataset.BeamSequence)}")
#     testa = plan_dataset.BeamSequence[beam_num]

#     testb = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum]
#     print(f"{cpnum} of {len(plan_dataset.BeamSequence[beam_num].ControlPointSequence)}")
   
#     print(f"Beam Limiting Sequence {len(plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence)}")
#     testc = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2]
    if len_sequence == 1:
        leftbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[0].LeafJawPositions[:num_leaves]
        rightbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[0].LeafJawPositions[num_leaves:]
        return leftbank,rightbank    
    
    elif len_sequence == 3:    
        leftbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions[:num_leaves]
        rightbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions[num_leaves:]
        return leftbank,rightbank
    
        #print("Cannot decode the form of this plan: BeamLimitingDevicePositionSequence error")
    else:
        return None

def get_leaf_positions_difference(plan_dataset,beam_num,cpnum,num_leaves):
    #leftbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions[:num_leaves]
    #rightbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions[num_leaves:]
    leftbank, rightbank = get_leaf_positions(plan_dataset,beam_num,cpnum)
    
    rightmat = np.array(rightbank)
    leftmat = np.array(leftbank) #np.flip(np.array(leftbank),0)
    diffmat = leftmat-rightmat
    return diffmat

def calc_control_point_aperture_area_jb(plan_dataset,beamno,cpnum,weight=1): 
    adjusted_width_matrix = calc_leaf_width_matrix(plan_dataset,beamno,cpnum,60,weight)    #maybe optimize, n-dim matrix ops
    cpaperturearea = np.sum(adjusted_width_matrix)                                                  #validate
    return cpaperturearea  

    
def get_control_point_aperture_perimeter(plan_dataset,beam_num,cpnum):
    lef, rig = get_leaf_positions(plan_dataset,beam_num,cpnum)
    
    lef = np.pad(lef,(1,1),'constant')
    rig = np.pad(rig,(1,1),'constant') #padding with zeros 
    
    lefdif, rightdif = np.diff(lef), np.diff(rig)
    prevPerim = np.sum(np.fabs(lefdif - rightdif))

    lefdifnext, rightdifnext = np.diff(np.flip(lef,0)), np.diff(np.flip(rig,0))
    nextPerim = np.sum(np.fabs(lefdifnext - rightdifnext))
    
    perim = prevPerim + nextPerim
    # Note:  np.diff returns out[n] = a[n+1] - a[n] 
    return perim



def get_cp_count(plan_dataset,beam_num):
    cpcount = plan_dataset.BeamSequence[beam_num].NumberOfControlPoints
    return cpcount

def get_MU_at_CP(plan_dataset,beam_num, cp_num):
    totalBeamMU = plan_dataset.FractionGroupSequence[0].ReferencedBeamSequence[beam_num].BeamMeterset
    if cp_num - 1 < 0:
        cp_num = 1 
    
    CulMetersetWeight = get_current_meterset_weight(plan_dataset,beam_num,cp_num)
    CulMetersetWeightPrev = get_current_meterset_weight(plan_dataset,beam_num,cp_num-1)
    MUatCP = totalBeamMU*(CulMetersetWeightPrev-CulMetersetWeight)
    return MUatCP

def get_current_meterset_weight(plan_dataset,beam_num,cp_num):
    culmeterweight = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cp_num].CumulativeMetersetWeight

    return culmeterweight

def calc_beam_aperture_complexity(plan_dataset,beamnum,verbose=False):
    MU = get_total_beam_MU(plan_dataset,beamnum)
    cpcount = get_cp_count(plan_dataset,beamnum)
    score = 0
    for i in range(0,cpcount):
        muatcp = get_MU_at_CP(plan_dataset,beamnum,i)
        perimatcp = get_control_point_aperture_perimeter(plan_dataset,beamnum,i)
        areaatcp = calc_control_point_aperture_area_jb(plan_dataset,beamnum,i)
        res = np.round(muatcp * ( perimatcp / areaatcp),5)/MU
        score += res
        
        if verbose == True:
            plt.scatter(i,res,marker=".",color="k")
            plt.ylim(0,0.005)
            plt.xlabel("Control Point")
            plt.ylabel("Aperture Complexity")
    
    study_score = score*100 #hardcoded to match ARIA version of score without explanation  
    return study_score


def get_field_sizes(plan_dataset,beamno):
    testmax = []
    

    for i in range(0,get_cp_count(plan_dataset,beamno)):
        len_sequence = len(plan_dataset.BeamSequence[beamno].ControlPointSequence[i].BeamLimitingDevicePositionSequence)
        if len_sequence == 3:
            ASYM_X = np.array(plan_dataset.BeamSequence[beamno].ControlPointSequence[i].BeamLimitingDevicePositionSequence[0].LeafJawPositions)
            ASYM_Y = np.array(plan_dataset.BeamSequence[beamno].ControlPointSequence[i].BeamLimitingDevicePositionSequence[1].LeafJawPositions)
        testmax.append([ASYM_X[1],-ASYM_X[0],ASYM_Y[1],-ASYM_Y[0]])
    testmax= np.array(testmax)
    fieldsizeX = np.amax(testmax[:,0]) + np.amax(testmax[:,1])
    fieldsizeY = np.amax(testmax[:,2]) + np.amax(testmax[:,3])
    #should probably switch this to an in-loop comparison. inefficient as is. 
    return fieldsizeX,fieldsizeY
    
    
def calc_harris_edge_score(dose_dataset):
    dose_edge_score = 0
    frame_no = len(dose_dataset.pixel_array)
    for i in range (0,frame_no):
        im = dose_dataset.pixel_array[i]
        coords = corner_peaks(corner_harris(im), min_distance=5)
        #coords_subpix = corner_subpix(im, coords, window_size=13)
        dose_edge_score += len(coords)
    normal_dose_edge = dose_edge_score/frame_no
    return normal_dose_edge
    
def calc_shape_index_scores(dose_dataset):
    shape_key = [-1,-7/8,-5/8,-3/8,-1/8,1/8,3/8,5/8,7/8,1]

    scores_list = [] 
    arr_size = len(dose_dataset.pixel_array)

    for j in range(0,len(shape_key)-1):
        tick = 0
        delta = 0.05


        for i in range (0,arr_size):
            im = dose_dataset.pixel_array[i]
            sh_im = shape_index(im)
            s_smooth = ndi.gaussian_filter(sh_im, sigma=0.875)

            point_y_s, point_x_s = np.where(  ( sh_im < shape_key[j+1])  & (sh_im >  shape_key[j]) )
            point_z_s = im[point_y_s, point_x_s]

            tick+=len(point_z_s)

        scores_list.append(np.around(tick/arr_size,3))
    return scores_list
    
def calc_shape_diff_scores(dose_dataset):
   shape_key = [-1,-7/8,-5/8,-3/8,-1/8,1/8,3/8,5/8,7/8,1]
   arr_list = []
   scores_list = [] 
   arr_size = len(dose_dataset.pixel_array)

   for i in range (0,arr_size):
       im = dose_dataset.pixel_array[i]
       sh_im = shape_index(im)
       arr_list.append(sh_im)
    
   score = np.abs(np.diff( np.nan_to_num(arr_list), axis=0).sum())/arr_size
    
        #scores_list.append(np.around(tick/arr_size,3))
   return score
   
def calc_CENSURE_score(dose_dataset):
    dose_score = 0
    frame_no = len(dose_dataset.pixel_array)
    frames = dose_dataset.pixel_array.copy()
    #pseudo_gray_frames = np.interp(frames, (frames.min(), frames.max()), (0, 255))
    for i in range (0,frame_no):
        im = frames[i]*255/frames.max()
        #im *= 255/frames.max()
        censure = CENSURE()
        censure.detect(im)
        key_points = censure.keypoints
        dose_score += len(key_points)
    normal_dose = dose_score/frame_no
    return normal_dose