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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASRESULTS_H
#define ASRESULTS_H

#include <asIncludes.h>
#include <asParameters.h>

class asFileNetcdf;


class asResults
        : public wxObject
{
public:
    asResults();

    virtual ~asResults();

    wxString GetPredictandStationIdsList();

    int GetCurrentStep()
    {
        return m_currentStep;
    }

    void SetCurrentStep(int val)
    {
        m_currentStep = val;
    }

    double GetDateProcessed()
    {
        return m_dateProcessed;
    }

    void SetDateProcessed(double val)
    {
        m_dateProcessed = val;
    }

    wxString GetFilePath()
    {
        return m_filePath;
    }

    void SetDateProcessed(const wxString &val)
    {
        m_filePath = val;
    }

    bool Exists();

    virtual bool Save(const wxString &AlternateFilePath = wxEmptyString);

    virtual bool Load(const wxString &AlternateFilePath = wxEmptyString);

protected:
    int m_fileVersionMajor;
    int m_fileVersionMinor;
    int m_currentStep;
    VectorInt m_predictandStationIds;
    double m_dateProcessed;
    wxString m_filePath;
    bool m_saveIntermediateResults;
    bool m_loadIntermediateResults;

    bool DefTargetDatesAttributes(asFileNetcdf &ncFile);

    bool DefStationIdsAttributes(asFileNetcdf &ncFile);

    bool DefStationOfficialIdsAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsNbAttributes(asFileNetcdf &ncFile);

    bool DefTargetValuesNormAttributes(asFileNetcdf &ncFile);

    bool DefTargetValuesGrossAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsCriteriaAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsDatesAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsValuesNormAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsValuesGrossAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsValuesAttributes(asFileNetcdf &ncFile);

    bool DefForecastScoresAttributes(asFileNetcdf &ncFile);

    bool DefForecastScoreFinalAttributes(asFileNetcdf &ncFile);

    bool DefLevelAttributes(asFileNetcdf &ncFile);

    bool DefScoresMapAttributes(asFileNetcdf &ncFile);

private:

};

#endif // ASRESULTS_H
