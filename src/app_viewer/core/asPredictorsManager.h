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

class asPredictorsManager {
  public:
    explicit asPredictorsManager(wxListBox* listPredictors, bool isTargetPredictor = false);

    virtual ~asPredictorsManager();

    asPredictor::Parameter GetParameter();

    bool LoadData();

    float* GetData();

    float* GetDataRow(int row);

    void SetDate(double date);

    int GetLongitudesNb() {
        return int(m_longitudes->size());
    }

    int GetLatitudesNb() {
        return int(m_latitudes->size());
    }

    double GetLongitudeMin() {
        return m_longitudes->minCoeff();
    }

    double GetLatitudeMax() {
        return m_latitudes->maxCoeff();
    }

    double GetLongitudeResol() {
        wxASSERT(m_longitudes->size() > 1);
        return (*m_longitudes)(1) - (*m_longitudes)(0);
    }

    double GetLatitudeResol() {
        wxASSERT(m_latitudes->size() > 1);
        return (*m_latitudes)(1) - (*m_latitudes)(0);
    }

  protected:
  private:
    wxListBox* m_listPredictors;
    bool m_isTargetPredictor;
    double m_date;
    a2f m_data;
    a1f m_longitudes;
    a1f m_latitudes;
};

#endif
