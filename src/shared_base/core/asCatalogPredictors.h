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
 
#ifndef ASCATALOGPREDICTOR_H
#define ASCATALOGPREDICTOR_H

#include <asIncludes.h>
#include <asCatalog.h>


class asCatalogPredictors: public asCatalog
{
public:

    //!< Structure for data information
    struct DataStruct
    {
        wxString Id;
        wxString Name;
        FileTimeLength FileLength;
        wxString FileName;
        wxString FileVarName;
        DataParameter Parameter;
        DataUnit Unit;
        double UaxisStep;
		double VaxisStep;
        double UaxisShift;
		double VaxisShift;
    };

    /** Default constructor
     * \param DataSetId The dataset ID
     * \param DataId The data ID. If not set, load the whole database information
     */
    asCatalogPredictors(const wxString &alternateFilePath = wxEmptyString);

    /** Default destructor */
    virtual ~asCatalogPredictors();

    virtual bool Load(const wxString &DataSetId, const wxString &DataId = wxEmptyString);

    /** Method that get dataset information from file
     * \param DataSetId The data Set ID
     */
    virtual bool LoadDatasetProp(const wxString &DataSetId);

    /** Method that get data information from file
     * \param DataSetId The data Set ID
     * \param DataId The data ID
     */
    virtual bool LoadDataProp(const wxString &DataSetId, const wxString &DataId);

    /** Access m_Data
     * \return The current value of m_Data
     */
    DataStruct GetDataInfo()
    {
        return m_Data;
    }

    /** Access m_Website
     * \return The current value of m_Website
     */
    wxString GetWebsite()
    {
        return m_Website;
    }

    /** Set m_Website
     * \param val New value to set
     */
    void SetWebsite(const wxString &val)
    {
        m_Website = val;
    }

    /** Access m_Ftp
     * \return The current value of m_Ftp
     */
    wxString GetFtp()
    {
        return m_Ftp;
    }

    /** Set m_Ftp
     * \param val New value to set
     */
    void SetFtp(const wxString &val)
    {
        m_Ftp = val;
    }

    /** Access m_Data.Id
     * \return The current value of m_Data.Id
     */
    wxString GetDataId()
    {
        return m_Data.Id;
    }

    /** Set m_Data.Id
     * \param val New value to set
     */
    void SetDataId(const wxString &val)
    {
        m_Data.Id = val;
    }

    /** Access m_Data.Name
     * \return The current value of m_Data.Name
     */
    wxString GetDataName()
    {
        return m_Data.Name;
    }

    /** Set m_Data.Name
     * \param val New value to set
     */
    void SetDataName(const wxString &val)
    {
        m_Data.Name = val;
    }

    /** Access m_Data.FileLength
     * \return The current value of m_Data.FileLength
     */
    FileTimeLength GetDataFileLength()
    {
        return m_Data.FileLength;
    }

    /** Set m_Data.FileLength
     * \param val New value to set
     */
    void SetDataFileLength(FileTimeLength &val)
    {
        m_Data.FileLength = val;
    }

    /** Access m_Data.FileName
     * \return The current value of m_Data.FileName
     */
    wxString GetDataFileName()
    {
        return m_Data.FileName;
    }

    /** Set m_Data.FileName
     * \param val New value to set
     */
    void SetDataFileName(const wxString &val)
    {
        m_Data.FileName = val;
    }

    /** Access m_Data.FileVarName
     * \return The current value of m_Data.FileVarName
     */
    wxString GetDataFileVarName()
    {
        return m_Data.FileVarName;
    }

    /** Set m_Data.FileVarName
     * \param val New value to set
     */
    void SetDataFileVarName(wxString val)
    {
        m_Data.FileVarName = val;
    }

    /** Access m_Data.Parameter
     * \return The current value of m_Data.Parameter
     */
    DataParameter GetDataParameter()
    {
        return m_Data.Parameter;
    }

    /** Set m_Data.Parameter
     * \param val New value to set
     */
    void SetDataParameter(DataParameter val)
    {
        m_Data.Parameter = val;
    }

    /** Access m_Data.Unit
     * \return The current value of m_Data.Unit
     */
    DataUnit GetDataUnit()
    {
        return m_Data.Unit;
    }

    /** Set m_Data.Unit
     * \param val New value to set
     */
    void SetDataUnit(DataUnit val)
    {
        m_Data.Unit = val;
    }

    /** Access m_Data.UaxisStep
     * \return The current value of m_Data.UaxisStep
     */
    double GetDataUaxisStep()
    {
        return m_Data.UaxisStep;
    }

    /** Set m_Data.UaxisStep
     * \param val New value to set
     */
    void SetDataUaxisStep(double val)
    {
        m_Data.UaxisStep = val;
    }

    /** Access m_Data.UaxisShift
     * \return The current value of m_Data.UaxisShift
     */
    double GetDataUaxisShift()
    {
        return m_Data.UaxisShift;
    }

    /** Set m_Data.UaxisShift
     * \param val New value to set
     */
    void SetDataUaxisShift(double val)
    {
        m_Data.UaxisShift = val;
    }

    /** Access m_Data.VaxisStep
     * \return The current value of m_Data.VaxisStep
     */
    double GetDataVaxisStep()
    {
        return m_Data.VaxisStep;
    }

    /** Set m_Data.VaxisStep
     * \param val New value to set
     */
    void SetDataVaxisStep(double val)
    {
        m_Data.VaxisStep = val;
    }

    /** Access m_Data.VaxisShift
     * \return The current value of m_Data.VaxisShift
     */
    double GetDataVaxisShift()
    {
        return m_Data.VaxisShift;
    }

    /** Set m_Data.VaxisShift
     * \param val New value to set
     */
    void SetDataVaxisShift(double val)
    {
        m_Data.VaxisShift = val;
    }



protected:
    wxString m_Website; //!< Member variable "m_Website"
    wxString m_Ftp; //!< Member variable "m_Ftp"
    DataStruct m_Data;  //!< Member variable "m_Data"


private:

};

#endif // ASCATALOGPREDICTOR_H
