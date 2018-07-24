# ======================== DATA PREPROCESSING ============================

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold # import KFold
import warnings
import numpy as np


# =========---------- File reading -------------=================
data = pd.read_csv("Subsets/TestingFewerResponses.csv", header=0)
df = pd.DataFrame(data)

print("")

warnings.filterwarnings('ignore')
# =============================== MAPPINGS ===================================

# ======== Mapping Marital Status to Integers ====================

label_encoder = LabelEncoder()
df["MaritalStatus"] = label_encoder.fit_transform(df["MaritalStatus"].fillna("NaN"))
# print(df.head(3))
# mapping = {label:idx for idx, label in enumerate(pd.unique(df["MaritalStatus"]))}
# print("MaritalStatus:", mapping)
# df["MaritalStatus"] = df["MaritalStatus"].map(mapping)
print("")

# ============= Mapping Employment Status to Integers ====================
#
df["EmploymentStatus"] = label_encoder.fit_transform(df["EmploymentStatus"].fillna("NaN"))

# =============== Mapping Housing Status to Integers ====================
#
df["HousingStatus"] = label_encoder.fit_transform(df["HousingStatus"].fillna("NaN"))

# =============== Mapping Type of Aid to Integers ====================

df["TypeOfAid"] = label_encoder.fit_transform(df["TypeOfAid"].astype(str).fillna("NaN"))

# # ============== Mapping ReasonForClosing to Integers ====================
#
df["ReasonForClosing"] = label_encoder.fit_transform(df["ReasonForClosing"].astype(str).fillna("NaN"))

print(label_encoder.classes_)

# # ============== Mapping NumberOfLiveBirths to Integers ====================
#
df["NumberOfLiveBirths"] = label_encoder.fit_transform(df["NumberOfLiveBirths"].astype(str).fillna("NaN"))

# # =============== Mapping N.NeonatalDeaths to Integers ====================
#
df["N.NeonatalDeaths"] = label_encoder.fit_transform(df["N.NeonatalDeaths"].astype(str).fillna("NaN"))

# # =============== Mapping N.LivingChildren to Integers ====================
#
df["N.LivingChildren"] = label_encoder.fit_transform(df["N.LivingChildren"].astype(str).fillna("NaN"))

# # =============== Mapping N.IncidentsPretermLabor to Integers ====================
#
df["N.IncidentsPretermLabor"] = label_encoder.fit_transform(df["N.IncidentsPretermLabor"].astype(str).fillna("NaN"))

# # ================ Mapping N.PretermDeliveries to Integers ====================
#
df["N.PretermDeliveries"] = label_encoder.fit_transform(df["N.PretermDeliveries"].astype(str).fillna("NaN"))

# # ================ Mapping N.LowBirthRateBabies to Integers ====================
#
df["N.LowBirthRateBabies"] = label_encoder.fit_transform(df["N.LowBirthRateBabies"].astype(str).fillna("NaN"))

# # ================ Mapping N.LaborComplications to Integers ====================
#
df["N.LaborComplications"] = label_encoder.fit_transform(df["N.LaborComplications"].astype(str).fillna("NaN"))

# # =================== Mapping HSDiplomaGED to Integers ===========================

df["HSDiplomaGED"] = label_encoder.fit_transform(df["HSDiplomaGED"].astype(str).fillna("NaN"))

# ============================== DROPPING INFORMATION ==========================================


print(df.shape)

y=df["ReasonForClosing"]
print()
x=df[["EmploymentStatus",
      "MaritalStatus",
      "HousingStatus",
      "TypeOfAid",
      "NumberOfLiveBirths",
      "N.NeonatalDeaths",
      "N.LivingChildren",
      "N.IncidentsPretermLabor",
      "N.PretermDeliveries",
      "N.LowBirthRateBabies",
      "N.LaborComplications",
      "HSDiplomaGED",
      "PsychiatricHistory",
      "ClientLevel"
      ]]

# print(x.head())
# x=df.drop("ReasonForClosing", axis=1)
# x=df.drop("ClientID", axis=1)
# x=df.drop("FamilyId", axis=1)
# x=df.drop("Visit Number", axis=1)
# x=df.drop("ClientLevel", axis=1)
# x=df.drop("PsychiatricHistory", axis=1)
# x=df.drop("TypeOfAid", axis=1)
# 77%
# x=df.drop("N.LaborComplications",axis=1)
# 80 %
# x=df.drop("N.LowBirthRateBabies",axis=1)
# 82%
# x=df.drop("N.PretermDeliveries",axis=1)
# 79%
# x=df.drop("N.IncidentsPretermLabor",axis=1)
# 84%
# x=df.drop("N.LivingChildren",axis=1)
# 79%
# x=df.drop("N.NeonatalDeaths",axis=1)
# 83%
# x=df.drop("NumberOfLiveBirths",axis=1)
# 77%
# x=df.drop("EmploymentStatus", axis=1)
# 78%
# x=df.drop("HousingStatus", axis=1)
# 82%
# x=df.drop("MaritalStatus", axis=1)

# print(x.describe)

# ================= SPLITTING THE DATA USING TRIAN-TEST-SPLIT MODEL ======================

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# print (x_train.shape, y_train.shape)
# print (x_test.shape, y_test.shape)
#
# scaler = StandardScaler()
# scaler.fit(x_train)
# # StandardScaler(copy=True, with_mean=True, with_std=True)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# =========== SPLITTING THE DATA USING K-FOLD CROSS VALIDATION MODEL =========================

kf = KFold( n_splits=4, random_state=None, shuffle=False)
kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator

for train_index, test_index in kf.split(x):
 # print('TRAIN:', train_index, 'TEST:', test_index)
 x_train, x_test = x.iloc[train_index], x.iloc[test_index]
 y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print("")
print ("x train data:", x_train.shape, y_train.shape)
print("")
print ("x test data:",x_test.shape, y_test.shape)
print("")

# ============================= SCALING ======================================

scaler = StandardScaler()
StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#

# ========================= MLP CLASSIFIER ======================================

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
model = mlp.fit(x_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(13, 13, 13), learning_rate='constant', learning_rate_init=0.001, max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)

# -----------------------------------------------------
predictions = mlp.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
confusion = confusion_matrix(y_test, predictions)
print(confusion)
print(classification_report(y_test, predictions))

## print(mlp.coefs_[0].max())
## print()

# =============================== PLOTTING THE RESULTS =======================================
# ========= SCATTER PLOT ==============
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
#
# plt.scatter(y_test, predictions, c='indigo' , marker='+')
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.show()
#
# print('Score:', model.score(x_test, y_test))
#

# ================ SCORES =====================
# print(mlp.n_iter_)
# print("Training set score: %f" % mlp.score(x_train, y_train))
# print("Test set score: %f" % mlp.score(x_test, y_test))



#  ================= CONFUSION MATRIX =============
from matplotlib.colors import LogNorm, ListedColormap

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

norm_conf = []
for i in confusion:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        if a == 0:
            tmp_arr.append(0)
        else:
            tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.viridis, interpolation='nearest')
# Hi(tler) -Jimmy
width, height = confusion.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(confusion[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = ('Billing issues', 'Child overage', 'Completed requirements',
 'Death of child', 'Gone back to school', 'Lost custody',
 'No longer wants services', 'Parent in rehab', 'Returned to school',
 'Returning to work', 'nan')
plt.xticks(range(width), alphabet[:width], rotation='vertical')
plt.yticks(range(height), alphabet[:height])
plt.subplots_adjust(bottom=0.15)
plt.show()
#
# # ========== COEFFICIENTS ================
#
# print(mlp.__getstate__())
fig, axes = plt.subplots(4, 3)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(1, 13), cmap=plt.cm.RdPu, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
# =========================================
