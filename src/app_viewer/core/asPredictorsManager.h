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

class asPredictorsManager {
  public:
    /**
     * The constructor of the class handling the predictor data for the Viewer.
     *
     * @param listPredictors The list of predictors from the interface.
     * @param workspace The opened workspace.
     * @param isTargetPredictor A boolean indicating if the predictor is the target.
     */
    explicit asPredictorsManager(wxListBox* listPredictors, asWorkspace* workspace, bool isTargetPredictor = false);

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
     * @return True if the data was loaded successfully.
     */
    bool LoadData();

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
        m_forecastDate = date;
        m_needsDataReload = true;
    }

    /**
     * Set the forecast time step for the predictor data.
     *
     * @param forecastTimeStepHours The forecast time step.
     */
    void SetForecastTimeStepHours(double forecastTimeStepHours) {
        m_forecastTimeStepHours = forecastTimeStepHours;
    }

    /**
     * Set the dataset IDs for the predictor data.
     *
     * @param predictorDatasetIds The dataset IDs.
     */
    void SetDatasetIds(const vwxs& predictorDatasetIds) {
        m_datasetIds = predictorDatasetIds;
        m_needsDataReload = true;
    }

    /**
     * Set the data IDs for the predictor data.
     *
     * @param predictorDataIds The data IDs.
     */
    void SetDataIds(const vwxs& predictorDataIds) {
        m_dataIds = predictorDataIds;
        m_needsDataReload = true;
    }

    /**
     * Set the vertical levels for the predictor data.
     *
     * @param predictorLevels The vertical levels.
     */
    void SetLevels(const vf& predictorLevels) {
        m_levels = predictorLevels;
        m_needsDataReload = true;
    }

    /**
     * Set the hours for the predictor data.
     *
     * @param predictorHours The hours.
     */
    void SetHours(const vf& predictorHours) {
        m_hours = predictorHours;
        m_needsDataReload = true;
    }

    /**
     * Get the number of longitude points.
     * 
     * @return The number of longitude points.
     */
    int GetLongitudesNb() {
        return int(m_longitudes->size());
    }

    /**
     * Get the number of latitude points.
     * 
     * @return The number of latitude points.
     */
    int GetLatitudesNb() {
        return int(m_latitudes->size());
    }

    /**
     * Get the minimum longitude value.
     * 
     * @return The minimum longitude value.
     */
    double GetLongitudeMin() {
        return m_longitudes->minCoeff();
    }

    /**
     * Get the minimum latitude value.
     * 
     * @return The minimum latitude value.
     */
    double GetLatitudeMin() {
        return m_latitudes->minCoeff();
    }

    /**
     * Get the maximum latitude value.
     * 
     * @return The maximum latitude value.
     */
    double GetLatitudeMax() {
        return m_latitudes->maxCoeff();
    }

    /**
     * Get the resolution of the longitude grid.
     * 
     * @return The resolution of the longitude grid.
     */
    double GetLongitudeResol() {
        wxASSERT(m_longitudes->size() > 1);
        return (*m_longitudes)(1) - (*m_longitudes)(0);
    }

    /**
     * Get the resolution of the latitude grid.
     * 
     * @return The resolution of the latitude grid.
     */
    double GetLatitudeResol() {
        wxASSERT(m_latitudes->size() > 1);
        return (*m_latitudes)(1) - (*m_latitudes)(0);
    }

    /**
     * Flag the need to reload the data.
     */
    void NeedsDataReload() {
        m_needsDataReload = true;
    }

  protected:
  private:
    asWorkspace* m_workspace; /**< The Viewer workspace. */
    wxListBox* m_listPredictors; /**< The list of predictors from the interface. */
    asPredictor* m_predictor; /**< The selected predictor. */
    bool m_isTargetPredictor; /**< A boolean indicating if the predictor is the target. */
    double m_forecastDate; /**< The forecast date as MJD. */
    double m_date; /**< The data date as MJD. */
    double m_forecastTimeStepHours;  /**< The forecast time step in hours. */
    vwxs m_datasetIds; /**< The dataset IDs. */
    vwxs m_dataIds; /**< The data IDs. */
    vf m_levels; /**< The vertical levels. */
    vf m_hours; /**< The hours. */
    bool m_needsDataReload; /**< A boolean indicating if the data needs to be reloaded. */
    a2f* m_data; /**< The loaded data. */
    a1d* m_longitudes; /**< The longitudes. */
    a1d* m_latitudes; /**< The latitudes. */
};

#endif
