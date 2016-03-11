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


class asResults: public wxObject
{
public:

    /** Default constructor */
    asResults();

    /** Default destructor */
    virtual ~asResults();

    wxString GetPredictandStationIdsList();

    /** Access m_currentStep
     * \return The current value of m_currentStep
     */
    int GetCurrentStep()
    {
        return m_currentStep;
    }

    /** Set m_currentStep
     * \param val The new value
     */
    void SetCurrentStep(int val)
    {
        m_currentStep = val;
    }

    /** Access m_dateProcessed
     * \return The current value of m_dateProcessed
     */
    double GetDateProcessed()
    {
        return m_dateProcessed;
    }

    /** Set m_dateProcessed
     * \param val The new value
     */
    void SetDateProcessed(double val)
    {
        m_dateProcessed = val;
    }

    /** Access m_dateProcessed
     * \return The current value of m_dateProcessed
     */
    wxString GetFilePath()
    {
        return m_filePath;
    }

    /** Set m_filePath
     * \param val The new value
     */
    void SetDateProcessed(const wxString& val)
    {
        m_filePath = val;
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
	int m_fileVersionMajor;
	int m_fileVersionMinor;
    int m_currentStep;
    VectorInt m_predictandStationIds;
    double m_dateProcessed; //!< Member variable "m_dateProcessed"
    wxString m_filePath; //!< Member variable "m_filePath"
    bool m_saveIntermediateResults;
    bool m_loadIntermediateResults;

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
