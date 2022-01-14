import pandas as pd
from difflib import SequenceMatcher


def main():
    # The two dataset Titles, uncomment the dataset you would like to use and comment the other.
    #title = "processed_sequencing_data_biochem1.csv"
    title = "processed_sequencing_data_biochem2_revised.csv"

    #  Loads the dataset.
    dataset = pd.read_csv(title)
    avg = 0

    # Calculates the number of Cycles in the dataset.
    loops = len(dataset.columns)//6

    print(title)

    # Evaluates each Cycle in the dataset.
    for i in range(1, loops+1):
        print('\nCycle ' + str(i) + ':')

        # Displays and counts the different pairs between the ref and call columns
        print(dataset.groupby(['ref_' + str(i), 'call_' + str(i)]).size().reset_index().rename(columns={0: 'count'}))

        # Creates a new column named ratio of 1's and 0's to find matches in ref and call
        dataset['ratio_' + str(i)] = dataset[['ref_' + str(i), 'call_'  + str(i)]].apply(
            lambda x: SequenceMatcher(None, x[0], x[1]).ratio(), axis=1)

        # Adds up the 1's in ratio to help calculate accuracy and error.
        accr = sum(dataset['ratio_' + str(i)]) / len(dataset.index)

        # Prints Accuracy, Error, and Amount of N values.
        print('Accuracy: ', accr * 100, '%')
        print('Error: ', (1 - accr) * 100, '%')
        print('Amount of N values: ', dataset['call_' + str(i)].value_counts().N)
        avg = accr + avg

    # Displays the average accuracy and error for the CSV.
    print('\nAverage Accuracy of Method: ', (avg * 100)/loops, '%')
    print('Average Error of Method: ', 100 - ((avg*100)/loops), '%')


if __name__ == "__main__":
    main()
