/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2022-2023 Pascal Horton, Terranum.
 */

#ifndef AS_PREDICTORS_MANAGER_H
#define AS_PREDICTORS_MANAGER_H

#include "asIncludes.h"
#include "asPredictor.h"
#include "asWorkspace.h"

/**
 * @brief The class handling the predictor data for the Viewer.
 *
 * This class is used to load the data for the selected predictor in the predictors frame (meteorological maps)
 */
class asPredictorsManager {
  public:
    /**
     * The constructor of the class handling the predictor data for the Viewer.
     *
     * @param workspace The opened workspace.
     * @param isTargetPredictor A boolean indicating if the predictor is the target.
     */
    explicit asPredictorsManager(asWorkspace* workspace, bool isTargetPredictor = false);

    /**
     * The destructor of the class handling the predictor data for the Viewer.
     */
    virtual ~asPredictorsManager();

    /**
     * Get the meteorological parameter in use.
     *
     * @return The meteorological parameter.
     */
    asPredictor::Parameter GetParameter();

    /**
     * Load the data for the selected predictor.
     *
     * @param selection The selection of the predictor in the list.
     * @return True if the data was loaded successfully.
     */
    bool LoadData(int selection);

    /**
     * Access to a pointer to the loaded data.
     *
     * @return A pointer to the loaded data.
     */
    float* GetData();

    /**
     * Access to a row of the loaded data array.
     *
     * @param row The row to access.
     * @return A row of the loaded data array.
     */
    float* GetDataRow(int row);

    /**
     * Get the minimum value of the loaded data.
     *
     * @return The minimum value of the loaded data.
     */
    float GetDataMin();

    /**
     * Get the maximum value of the loaded data.
     *
     * @return The maximum value of the loaded data.
     */
    float GetDataMax();

    /**
     * Set the desired date for the predictor data.
     *
     * @param date The desired date.
     */
    void SetDate(double date);

    /**
     * Set the desired forecast date for the predictor data.
     *
     * @param date The desired forecast date.
     */
    void SetForecastDate(double date) {
        _forecastDate = date;
        _needsDataReload = true;
    }

    /**
     * Set the forecast time step for the predictor data.
     *
     * @param forecastTimeStepHours The forecast time step.
     */
    void SetForecastTimeStepHours(double forecastTimeStepHours) {
        _forecastTimeStepHours = forecastTimeStepHours;
    }

    /**
     * Set the dataset IDs for the predictor data.
     *
     * @param predictorDatasetIds The dataset IDs.
     */
    void SetDatasetIds(const vwxs& predictorDatasetIds) {
        _datasetIds = predictorDatasetIds;
        _needsDataReload = true;
    }

    /**
     * Set the data IDs for the predictor data.
     *
     * @param predictorDataIds The data IDs.
     */
    void SetDataIds(const vwxs& predictorDataIds) {
        _dataIds = predictorDataIds;
        _needsDataReload = true;
    }

    /**
     * Set the vertical levels for the predictor data.
     *
     * @param predictorLevels The vertical levels.
     */
    void SetLevels(const vf& predictorLevels) {
        _levels = predictorLevels;
        _needsDataReload = true;
    }

    /**
     * Set the hours for the predictor data.
     *
     * @param predictorHours The hours.
     */
    void SetHours(const vf& predictorHours) {
        _hours = predictorHours;
        _needsDataReload = true;
    }

    /**
     * Get the number of longitude points.
     * 
     * @return The number of longitude points.
     */
    int GetLongitudesNb() {
        return int(_longitudes->size());
    }

    /**
     * Get the number of latitude points.
     * 
     * @return The number of latitude points.
     */
    int GetLatitudesNb() {
        return int(_latitudes->size());
    }

    /**
     * Get the minimum longitude value.
     * 
     * @return The minimum longitude value.
     */
    double GetLongitudeMin() {
        return _longitudes->minCoeff();
    }

    /**
     * Get the minimum latitude value.
     * 
     * @return The minimum latitude value.
     */
    double GetLatitudeMin() {
        return _latitudes->minCoeff();
    }

    /**
     * Get the maximum latitude value.
     * 
     * @return The maximum latitude value.
     */
    double GetLatitudeMax() {
        return _latitudes->maxCoeff();
    }

    /**
     * Get the resolution of the longitude grid.
     * 
     * @return The resolution of the longitude grid.
     */
    double GetLongitudeResol() {
        wxASSERT(_longitudes->size() > 1);
        return (*_longitudes)(1) - (*_longitudes)(0);
    }

    /**
     * Get the resolution of the latitude grid.
     * 
     * @return The resolution of the latitude grid.
     */
    double GetLatitudeResol() {
        wxASSERT(_latitudes->size() > 1);
        return (*_latitudes)(1) - (*_latitudes)(0);
    }

    /**
     * Flag the need to reload the data.
     */
    void NeedsDataReload() {
        _needsDataReload = true;
    }

  protected:
  private:
    asWorkspace* _workspace; /**< The Viewer workspace. */
    asPredictor* _predictor; /**< The selected predictor. */
    bool _isTargetPredictor; /**< A boolean indicating if the predictor is the target. */
    double _forecastDate; /**< The forecast date as MJD. */
    double _date; /**< The data date as MJD. */
    double _forecastTimeStepHours;  /**< The forecast time step in hours. */
    vwxs _datasetIds; /**< The dataset IDs. */
    vwxs _dataIds; /**< The data IDs. */
    vf _levels; /**< The vertical levels. */
    vf _hours; /**< The hours. */
    bool _needsDataReload; /**< A boolean indicating if the data needs to be reloaded. */
    a2f* _data; /**< The loaded data. */
    a1d* _longitudes; /**< The longitudes. */
    a1d* _latitudes; /**< The latitudes. */
};

#endif
