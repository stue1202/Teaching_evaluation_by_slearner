import pickle
import sys
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from causalml.inference.meta import BaseSRegressor
from IPython.core import ultratb
from lightgbm import LGBMRegressor
from scipy import stats

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

warnings.filterwarnings("ignore")

with open("CID.pkl", "rb") as f:
    data_cid = pickle.load(f)

with open("SID.pkl", "rb") as f:
    data_sid = pickle.load(f)


def get_data(year, course, log=False):
    year = year - 108
    assert year > 0
    sids = [sid for sid in data_cid[year][course].keys()]
    cids = {cid for sid in sids for cid in data_sid[year - 1].get(sid, {}).keys()}
    x = {}
    students = []
    # if log:
    #     print("n student:", len(sids))
    for cid in cids:
        cnt = 0
        x[cid] = []
        for sid in sids:
            score = data_cid[year - 1][cid].get(sid, None)
            if score is not None:
                cnt += 1
            x[cid].append(score)

        if cnt / len(sids) < 0.1:
            x.pop(cid)
        else:
            #print(cnt)
            students.append(cnt)

    y = [data_cid[year][course][sid] for sid in sids]

    #if log:
        #print("cids:", list(x.keys()))

    feature_name = list(x.keys())
    x = np.array(list(x.values()), dtype=np.float)
    x = np.swapaxes(x, 0, 1)
    y = np.nan_to_num(np.array(y))

    return x, y, feature_name, students


def inference(x, t, y):
    model = LGBMRegressor(verbose=-1)
    slearner = BaseSRegressor(model, control_name="control")
    result = slearner.estimate_ate(x, t, y)
    return result

c_t108 = {}
teachers108 = {}
with open('108上學期全校課程資訊.pkl', 'rb') as f:
    dic1081 = pickle.load(f)
with open('108下學期全校課程資訊.pkl', 'rb') as f:
    dic1082 = pickle.load(f)
def create_map():
    #print(dic1081)
    # print(dic1081.keys())
    # print(dic1082.keys())
    for course in dic1081:
        #print(type(course))
        if ('CE' in course and len(dic1081[course]['course_teacher']) <= 3):
            #print(course, dic1081[course]['course_teacher'])
            c_t108[course] = dic1081[course]['course_teacher']
            teachers108[dic1081[course]['course_teacher']] = {'score': 0, 'students': 0}

    for course in dic1082:
        #print(type(course))
        if ('CE' in course and len(dic1082[course]['course_teacher']) <= 3):
            #print(course, dic1082[course]['course_teacher'])
            c_t108[course] = dic1082[course]['course_teacher']
            teachers108[dic1082[course]['course_teacher']] = {'score': 0, 'students': 0}
    
    i = 1
    for teacher in teachers108:
        teachers108[teacher]['num'] = i
        i += 1

if __name__ == "__main__":
    create_map()
    print(c_t108)
    print(teachers108)

    get_data_error = []
    inference_error = []
    key_error = []
    ab_error = []
    output = []

    for course in c_t108:
        if(course in dic1081):
            print(dic1081[course]['course_name'])
        elif (course in dic1082):
            print(dic1082[course]['course_name'])
        output.append(course)

        if('-*' in course):
            course = course[:-2]
        target_cid = course
        #target_cid = "CE1001"  # CE3006, CE2005
        try:
            x, y, feature_name, students = get_data(111, target_cid, log=True)
        except KeyError:
            if('-' not in course):
                course += '-*'
            if(course in dic1081):
                #print('get_data error:' + dic1081[course]['course_name'])
                key_error.append(dic1081[course]['course_name'])
            elif (course in dic1082):
                #print('get_data error:' + dic1082[course]['course_name'])
                key_error.append(dic1082[course]['course_name'])
            else:
                key_error.append(course)
            
            
            # if '*' in course:
            #     course = course[:-1] + 'A'
            #     target_cid = course
            #     try:
            #         x, y, feature_name, students = get_data(111, target_cid, log=True)
            #     except:
            #         print('wow')
            # else:
            #     course = course[:-2]
            #     target_cid = course
            #     try:
            #         x, y, feature_name, students = get_data(111, target_cid, log=True)
            #     except:
            #         print('wow')

            try:
                if '*' in course:
                    course = course[:-1] + 'B'
                else:
                    course = course[:-2]
                target_cid = course
                x, y, feature_name, students = get_data(111, target_cid, log=True)
            except:
                ab_error.append(course)

        except:
            if('-' not in course):
                course += '-*'
            if(course in dic1081):
                #print('get_data error:' + dic1081[course]['course_name'])
                get_data_error.append(dic1081[course]['course_name'])
            elif (course in dic1082):
                #print('get_data error:' + dic1082[course]['course_name'])
                get_data_error.append(dic1082[course]['course_name'])
            else:
                get_data_error.append(course)
            continue

        ates = []
        for i in range(len(feature_name)):
            t = np.where(np.isnan(x[:, i]), "control", "treatment")
            _x = np.nan_to_num(np.delete(x, i, 1))
            try:
                ates.append(max(1e-4, inference(_x, t, y)[0]))
            except:
                if('-' not in course):
                    course += '-*'
                if(course in dic1081):
                    #print('get_data error:' + dic1081[course]['course_name'])
                    inference_error.append(dic1081[course]['course_name'])
                elif (course in dic1082):
                    #print('get_data error:' + dic1082[course]['course_name'])
                    inference_error.append(dic1082[course]['course_name'])
                else:
                    inference_error.append(course)
            #print(feature_name[i], ates[-1])

        ts = []
        mu = np.mean(y)
        for i in range(len(feature_name)):
            q = np.logical_not(np.isnan(x[:, i]))
            p = np.corrcoef(q, y)[0, 1]
            ts.append(p)
            # print(feature_name[i], t)

        cid2name = {}
        with open("./cid.txt", "r", encoding='utf-8') as f:
            for s in f.readlines():
                cid = s.split()[0]
                cname = " ".join(s.split()[1:])
                cid2name[cid] = cname

        # def draw(x_name, values):
        #     results = {}
        #     for cid, v in zip(feature_name, values):
        #         if "-" in cid:
        #             results[cid2name[cid] + "-" + cid.split("-")[1]] = v
        #     results = dict(reversed(sorted(results.items())))
        #     # print(feature_name)

        #     print(x_name)
        #     for k, v in results.items():
        #         print(k, v)

        #     import plotly.express as px

        #     target_name = cid2name.get(target_cid, None) or cid2name[target_cid + "-A"]
        #     fig = px.histogram(
        #         title=target_name, y=list(results.keys()), x=list(results.values())
        #     )
        #     fig.update_layout(
        #         xaxis_title_text=x_name,
        #         yaxis_title_text="Course",
        #         title_x=0.5,
        #         title_y=0.8,
        #         width=700,
        #         height=400,
        #     )
        #     fig.show()

        #draw("Treatment effect", ates)
        #draw("Correlation", ts)
        
        print(feature_name)
        print(ates)
        print(students)
        output.append(feature_name)
        output.append(ates)
        output.append(students)

        for i in range(len(ates)):
            if '-' not in feature_name[i]:
                feature_name[i] += '-*'

            if ates[i] > 0.01 and (feature_name[i] in c_t108):
                #print(feature_name[i])
                teachers108[c_t108[feature_name[i]]]['score'] += ates[i] * students[i]
                teachers108[c_t108[feature_name[i]]]['students'] += students[i]

            if feature_name[i] in c_t108:
                print(feature_name[i], "Professor" + str(teachers108[c_t108[feature_name[i]]]['num']), teachers108[c_t108[feature_name[i]]])
                output.append(feature_name[i])
                output.append(c_t108[feature_name[i]])
                output.append(teachers108[c_t108[feature_name[i]]])
        print()

    # with open('output.txt', 'w') as f:
    #     f.write(output)

    print('key_error:')
    print(key_error)
    print('ab_error:')
    print(ab_error)
    print('get_data_error:')
    print(get_data_error)
    print('inference_error:')
    print(inference_error)
    print()

    for info in teachers108:
        if teachers108[info]['students'] > 0:
            print("Professor" + str(teachers108[info]['num']) + ":", float(teachers108[info]['score'] / teachers108[info]['students']))
        #print(info, teachers108[info]['num'])
            
    