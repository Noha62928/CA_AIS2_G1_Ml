import streamlit as st 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
st.set_page_config(
    page_title="Boston Housing Price Prediction",
    layout="wide"
)
st.markdown("""
<style>

    /* Main page background */
    .stApp {
        background-color: #F5F7FA;
    }

    /* Main title */
    h1 {
        color: #12355B;
        font-size: 42px;
        font-weight: 700;
    }

    /* Section headings */
    h2 {
        color: #1F5F8B;
    }

    h3 {
        color: #2878B5;
    }

    /* Normal text */
    p {
        color: #333333;
        font-size: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #12355B;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    }

    /* Buttons */
    .stButton > button {
        background-color: #2878B5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #12355B;
        color: white;
    }

</style>
""", unsafe_allow_html=True)
st.title("Boston House Price Prediction")
st.write("Welcome!")
name = st.text_input("What is your name?") 
if st.button("Submit"): 
    st.write("Hello", name)
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")
data = load_data()
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Overview",
        "Statistics",
        "Model Training",
        "Prediction"
    ]
)

prices = data["MEDV"]

features = data.drop("MEDV", axis=1)

selected_features = ["RM", "LSTAT", "PTRATIO"]

X = features[selected_features]
y = prices


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1
)


@st.cache_resource
def train_model(X_train, y_train):

    cv_sets = ShuffleSplit(
        n_splits=10,
        test_size=0.20,
        random_state=0
    )

    regressor = DecisionTreeRegressor(
        random_state=0
    )

    params = {
        "max_depth": list(range(1, 11))
    }

    scoring_fnc = make_scorer(r2_score)

    grid = GridSearchCV(
        estimator=regressor,
        param_grid=params,
        scoring=scoring_fnc,
        cv=cv_sets
    )

    grid.fit(X_train, y_train)

    return grid


grid = train_model(X_train, y_train)

reg = grid.best_estimator_

y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


if page == "Overview":

    st.header("Overview")

    st.write("First 5 rows of the dataset:")

    st.dataframe(
        data.head()
    )

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Number of Rows",
            data.shape[0]
        )

    with col2:
        st.metric(
            "Number of Columns",
            data.shape[1]
        )

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str)
    })

    st.dataframe(dtype_df)



elif page == "Statistics":

    st.header("Statistics")

    st.subheader("Descriptive Statistics")

    st.dataframe(
        data.describe()
    )

    st.subheader("MEDV Statistics")

    medv_stats = data["MEDV"].describe()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Minimum Price is",
            f"${data['MEDV'].min():,.2f}"
        )

    with col2:
        st.metric(
            "Maximum Price is",
            f"${data['MEDV'].max():,.2f}"
        )

    with col3:
        st.metric(
            "Mean Price is",
            f"${data['MEDV'].mean():,.2f}"
        )

    with col4:
        st.metric(
            "Median Price is",
            f"${data['MEDV'].median():,.2f}"
        )

    st.subheader("Standard Deviation")

    st.write(
        f"${data['MEDV'].std():,.2f}"
    )

    st.subheader("Missing Values")

    missing = data.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(missing_df)

    st.subheader("Selected Features")

    st.write(
        """
        The model uses:

        - **RM**
        - **LSTAT**
        - **PTRATIO**
        """
    )

    st.subheader("Feature Relationships")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("RM vs MEDV")
        st.scatter_chart(
            data[["RM", "MEDV"]]
        )

    with col2:
        st.write("LSTAT vs MEDV")
        st.scatter_chart(
            data[["LSTAT", "MEDV"]]
        )

    with col3:
        st.write("PTRATIO vs MEDV")
        st.scatter_chart(
            data[["PTRATIO", "MEDV"]]
        )


elif page == "Model Training":

    st.header("Model Training")


    st.subheader("Training / Testing Split")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Samples",
            len(X_train)
        )

    with col2:
        st.metric(
            "Testing Samples",
            len(X_test)
        )

    st.subheader("Optimal Model")

    st.success(
        f"Best max_depth = {reg.get_params()['max_depth']}"
    )

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    with col2:
        st.metric(
            "MAE",
            f"{mae:.3f}"
        )

    with col3:
        st.metric(
            "RMSE",
            f"{rmse:.3f}"
        )

    st.subheader("Grid Search Results")

    results = pd.DataFrame(grid.cv_results_)

    results_display = results[
        [
            "param_max_depth",
            "mean_test_score",
            "std_test_score"
        ]
    ]

    results_display.columns = [
        "Max Depth",
        "Mean R²",
        "Std R²"
    ]

    st.dataframe(
        results_display
    )

    st.subheader("Validation Performance")

    chart_data = results_display.copy()

    chart_data["Max Depth"] = chart_data["Max Depth"].astype(int)

    chart_data = chart_data.set_index("Max Depth")

    st.line_chart(
        chart_data["Mean R²"]
    )


elif page == "Prediction":

    st.header("House Price Prediction")

    st.write(
        """
        Enter the characteristics of the house below.
        The trained Decision Tree model will estimate its price.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        rm = st.number_input(
            "Average number of rooms (RM)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1
        )

    with col2:

        lstat = st.number_input(
            "Neighborhood poverty level (LSTAT %)",
            min_value=0.0,
            max_value=100.0,
            value=17.0,
            step=0.1
        )

    with col3:

        ptratio = st.number_input(
            "Student-teacher ratio (PTRATIO)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.1
        )

    if st.button("Predict Price"):

        client = pd.DataFrame(
            {
                "RM": [rm],
                "LSTAT": [lstat],
                "PTRATIO": [ptratio]
            }
        )

        prediction = reg.predict(client)[0]

        st.success(
            f"Estimated Selling Price: ${prediction:,.2f}"
        )

        st.subheader("Input Data")

        st.dataframe(client)

        st.subheader("Model Information")

        st.write(
            f"Optimal Decision Tree max_depth: "
            f"**{reg.get_params()['max_depth']}**"
        )

        st.write(
            f"Model R² score on test data: "
            f"**{r2:.3f}**"
        )



st.sidebar.markdown("---")

st.sidebar.info(
    "Boston Housing Price Prediction\n\n"
)