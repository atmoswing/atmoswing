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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASRESULTS_H
#define ASRESULTS_H

#include <asIncludes.h>
#include <asParameters.h>

class asFileNetcdf;


class asResults: public wxObject
{
public:

    /** Default constructor */
    asResults();

    /** Default destructor */
    virtual ~asResults();

    wxString GetPredictandStationIdsList();

    /** Access m_CurrentStep
     * \return The current value of m_CurrentStep
     */
    int GetCurrentStep()
    {
        return m_CurrentStep;
    }

    /** Set m_CurrentStep
     * \param val The new value
     */
    void SetCurrentStep(int val)
    {
        m_CurrentStep = val;
    }

    /** Access m_DateProcessed
     * \return The current value of m_DateProcessed
     */
    double GetDateProcessed()
    {
        return m_DateProcessed;
    }

    /** Set m_DateProcessed
     * \param val The new value
     */
    void SetDateProcessed(double val)
    {
        m_DateProcessed = val;
    }

    /** Access m_DateProcessed
     * \return The current value of m_DateProcessed
     */
    wxString GetFilePath()
    {
        return m_FilePath;
    }

    /** Set m_FilePath
     * \param val The new value
     */
    void SetDateProcessed(const wxString& val)
    {
        m_FilePath = val;
    }

    /** Check if the result file exists
     * \return True if exists
     */
    bool Exists();

    /** Save the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    virtual bool Save(const wxString &AlternateFilePath = wxEmptyString);

    /** Load the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    virtual bool Load(const wxString &AlternateFilePath = wxEmptyString);

protected:
    float m_FileVersion;
    int m_CurrentStep;
    VectorInt m_PredictandStationIds;
    double m_DateProcessed; //!< Member variable "m_DateProcessed"
    wxString m_FilePath; //!< Member variable "m_FilePath"
    bool m_SaveIntermediateResults;
    bool m_LoadIntermediateResults;

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefTargetDatesAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefStationIdsAttributes(asFileNetcdf &ncFile);
    bool DefStationOfficialIdsAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefAnalogsNbAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefTargetValuesNormAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefTargetValuesGrossAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefAnalogsCriteriaAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefAnalogsDatesAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefAnalogsValuesNormAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefAnalogsValuesGrossAttributes(asFileNetcdf &ncFile);

    bool DefAnalogsValuesAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefForecastScoresAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefForecastScoreFinalAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefLevelAttributes(asFileNetcdf &ncFile);

    /** Defines attributes in the destination file
     * \param ncFile The NetCDF file to save the result
     * \return True on success
     */
    bool DefScoresMapAttributes(asFileNetcdf &ncFile);

private:

};

#endif // ASRESULTS_H
