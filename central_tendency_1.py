def mean(data): 
    return sum(data)/len(data) 
 
def median(data): 
    n=len(data) 
    mid=n//2 
    sort_data=sorted(data) 
    if n%2==0: 
        return (sort_data[mid-1]+sort_data[mid])/2 
    else: 
        return sort_data[mid] 
 
def mode(data): 
    frequency={} 
    for num in data: 
        frequency[num]=frequency.get(num,0)+1 
    max_freq=max(frequency.values()) 
    modes=[key for key,value in frequency.items() if value==max_freq] 
    return modes 
 
def variance(data): 
    Mean=mean(data) 
    sq_diff=[(x-Mean)**2 for x in data] 
    return sum(sq_diff)/(len(data)-1) 
 
def std_dev(data): 
    return variance(data)**0.5 
 
def statistics(data): 
    print("Data:",data) 
    Mean=mean(data) 
    Median=median(data) 
    Mode=mode(data) 
    Variance=variance(data) 
    Std_dev=std_dev(data) 
    print(f"Mean={Mean}") 
    print(f"Median={Median}") 
    print(f"Mode={Mode}") 
    print(f"Variance={Variance}") 
    print(f"Std_dev={Std_dev}") 
 
if __name__=="__main__": 
    user_input=input("Enter the numbers seperated by spaces:") 
    sample_data=list(map(float,user_input.strip().split())) 
    statistics(sample_data)