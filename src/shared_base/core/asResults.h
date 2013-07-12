/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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
    int m_PredictandStationId;
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
    bool DefStationsIdsAttributes(asFileNetcdf &ncFile);

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
    bool DefLonLatAttributes(asFileNetcdf &ncFile);

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
