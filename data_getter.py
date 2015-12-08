import os

import numpy as np
import sklearn.preprocessing as pp
from xlrd import open_workbook

__author__ = 'MagnieAr'


def get_returns(sht_idx, normalize, observations_nbr):

    file_path=r'P:\Projects\Master Thesis'

    Book = open_workbook(os.path.join(file_path,"XHRC-datareduced.xlsm"), on_demand=True)

    # Control parameters

    sheet_name = Book.sheet_by_index(sht_idx).name
    sheet = Book.sheet_by_name(sheet_name)

    # num_rows = sheet.nrows - 1
    num_rows = observations_nbr
    # num_rows = 120/220
    num_cols = sheet.ncols - 1

    all_maturities = [0 for x in range(num_rows)]

    scatter_matrix = np.zeros((num_cols, num_rows))
    scatter_returns = np.zeros((num_cols, num_rows-1))

    for col_idx in range(1, num_cols+1):
            for row_idx in range(1, num_rows+1):
                    cell_obj = sheet.cell(row_idx, col_idx)
                    all_maturities[row_idx-1] = cell_obj.value
            scatter_matrix[col_idx-1] = all_maturities

    for col_idx in range(0, num_cols):  # iterates on time-to-maturities
        for row_idx in range(0, num_rows):  # Iterates on observations
                all_maturities[row_idx] = scatter_matrix.T[row_idx][col_idx]  # Store the corresponding price
                if row_idx > 0:  # Compute the returns and store it in a num_cols x num_rows matrix
                    scatter_returns[col_idx][row_idx-1] = (all_maturities[row_idx] - all_maturities[row_idx-1]) / \
                                                          all_maturities[row_idx-1]

    # R = scatter_returns.T
    if normalize:
        R = pp.normalize(scatter_returns.T, norm='l2', axis=0, copy=True)
    else:
        R = scatter_returns.T


    return R, sheet_name;


def get_forward_curve(commo, maturity):

    file_path=r'P:\Projects\Master Thesis'

    Book = open_workbook(os.path.join(file_path,"XACO.xlsm"), on_demand=True)

    sheet_name = Book.sheet_by_index(0).name
    sheet = Book.sheet_by_name(sheet_name)

    forward_curve = []

    cell_obj_name = sheet.cell(maturity+1, commo*2)
    cell_obj_price = sheet.cell(maturity+1, commo*2 + 1)
    forward_curve.append((cell_obj_name.value, cell_obj_price.value))

    return forward_curve, sheet.cell(1, commo*2 + 1).value;


def get_strikes(commo):

    file_path=r'P:\Projects\Master Thesis'

    Book = open_workbook(os.path.join(file_path,"XACO.xlsm"), on_demand=True)

    sheet_name = Book.sheet_by_index(0).name
    sheet = Book.sheet_by_name(sheet_name)

    strikes = []

    val = ""
    i = 0
    while val != "-":
        cell_obj_name = sheet.cell(20+i, commo*2 + 1)
        i += 1
        val = cell_obj_name.value
        strikes.append(val)

    return strikes;