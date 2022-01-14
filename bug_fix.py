import pandas as pd


def main():
    # Loads in bugged dataset.
    dataset = pd.read_csv("sequencing_data_biochem2.csv")
    dataset_copy = dataset.copy()

    # Checks for correlations in ref and call columns.
    corr = dataset.groupby(['ref_1','call_1']).size().reset_index().rename(columns={0:'count'})

    # Sorts in descending order.
    corr = corr.sort_values(by='count', ascending=False)
    corr = corr[0:4]
    print(corr)
    print(dataset)

    # Migrates data to the proper columns.
    for i in range(1,3):
        dataset['A_' + str(i)], dataset['C_' + str(i)], dataset['G_' + str(i)], dataset['T_' + str(i)] = dataset_copy['T_' + str(i)], dataset_copy['A_' + str(i)], dataset_copy['C_' + str(i)], dataset_copy['A_' + str(i)]

    print(dataset)

    # Exports revised CSV.
    dataset.to_csv('sequencing_data_biochem2_revised.csv', index=False)


if __name__ == "__main__":
    main()
