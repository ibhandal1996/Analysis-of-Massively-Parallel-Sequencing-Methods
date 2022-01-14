import tensorflow as tf
import pandas as pd


# Predicts the values for call using the model.
def predict(model, A, C, G, T):
  example = tf.train.Example()
  example.features.feature["A_1"].float_list.value.extend([A])
  example.features.feature["C_1"].float_list.value.extend([C])
  example.features.feature["G_1"].float_list.value.extend([G])
  example.features.feature["T_1"].float_list.value.extend([T])
  return model.signatures["predict"](
    examples=tf.constant([example.SerializeToString()]))


def main():
    # Loads the model.
    model = tf.saved_model.load('myModel\\1636702458')
    # The two dataset Titles, uncomment the dataset you would like to use and comment the other.
    #title = "sequencing_data_biochem1.csv"
    title = "sequencing_data_biochem2_revised.csv"
    x = pd.read_csv(title)

    # Calculates the number of Cycles in the dataset for the model and drops the call columns.
    float_values = int(x.dtypes.value_counts()[0])
    loops = float_values // 4
    x = x[x.columns.drop(list(x.filter(regex='call')))]

    call = pd.DataFrame()

    # Creates a dictionary for converting the number values to A, C, G, T, N values
    ACGTN = {
        0: 'A',
        1: 'C',
        2: 'G',
        3: 'T',
        4: 'N'
        }

    asd = []

    # Loops through each cycle for predictions from model.
    for j in range(1, loops + 1):
        for i in x.index:
            pr = predict(model, x['A_' + str(j)][i], x['C_' + str(j)][i], x['G_' + str(j)][i], x['T_' + str(j)][i])
            pr = int(pr['class_ids'].numpy())
            asd.append(pr)
            call.loc[i, 'call_' + str(j)] = pr
        # Coverts the number values to A, C, G, T, N values
        call['call_' + str(j)] = call['call_' + str(j)].map(ACGTN)
        # Adds the values of call to the dataframe, which will be our output CSV.
        x.insert(x.columns.get_loc('T_' + str(j)) + 1, 'call_' + str(j), call['call_' + str(j)])

    # Exports processed CSV.
    x.to_csv('processed_' + title, index=False)


if __name__ == "__main__":
    main()
