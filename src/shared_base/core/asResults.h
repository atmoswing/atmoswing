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

    wxString GetPredictandStationIdsList() const;

    int GetCurrentStep() const
    {
        return m_currentStep;
    }

    void SetCurrentStep(int val)
    {
        m_currentStep = val;
    }

    double GetDateProcessed() const
    {
        return m_dateProcessed;
    }

    void SetDateProcessed(double val)
    {
        m_dateProcessed = val;
    }

    wxString GetFilePath() const
    {
        return m_filePath;
    }

    void SetFilePath(const wxString &val)
    {
        m_filePath = val;
    }

    void SetSubFolder(const wxString &val)
    {
        m_subFolder = val;
    }

    bool Exists() const;

    virtual bool Save();

    virtual bool Load();

protected:
    int m_fileVersionMajor;
    int m_fileVersionMinor;
    int m_currentStep;
    vi m_predictandStationIds;
    double m_dateProcessed;
    wxString m_subFolder;
    wxString m_filePath;

    bool DefTargetDatesAttributes(asFileNetcdf &ncFile) const;

    bool DefStationIdsAttributes(asFileNetcdf &ncFile) const;

    bool DefStationOfficialIdsAttributes(asFileNetcdf &ncFile) const;

    bool DefAnalogsNbAttributes(asFileNetcdf &ncFile) const;

    bool DefTargetValuesNormAttributes(asFileNetcdf &ncFile) const;

    bool DefTargetValuesGrossAttributes(asFileNetcdf &ncFile) const;

    bool DefAnalogsCriteriaAttributes(asFileNetcdf &ncFile) const;

    bool DefAnalogsDatesAttributes(asFileNetcdf &ncFile) const;

    bool DefAnalogsValuesNormAttributes(asFileNetcdf &ncFile) const;

    bool DefAnalogsValuesGrossAttributes(asFileNetcdf &ncFile) const;

    bool DefAnalogsValuesAttributes(asFileNetcdf &ncFile) const;

    bool DefScoresAttributes(asFileNetcdf &ncFile) const;

    bool DefTotalScoreAttributes(asFileNetcdf &ncFile) const;

    bool DefLevelAttributes(asFileNetcdf &ncFile) const;

    bool DefScoresMapAttributes(asFileNetcdf &ncFile) const;

private:

};

#endif // ASRESULTS_H
