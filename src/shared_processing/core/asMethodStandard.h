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
 
#ifndef ASMETHODSTANDARD_H
#define ASMETHODSTANDARD_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asMethodStandard: public wxObject
{
public:
    /** Default constructor */
    asMethodStandard();
    /** Default destructor */
    virtual ~asMethodStandard();

    void Cleanup();

    virtual bool Manager();

    bool LoadPredictandDB(const wxString &predictandDBFilePath);

    void Cancel();

    /** Access m_ParamsFilePath
     * \return The current value of m_ParamsFilePath
     */
    wxString GetParamsFilePath()
    {
        return m_ParamsFilePath;
    }

    /** Set m_ParamsFilePath
     * \param val New value to set
     */
    void SetParamsFilePath(const wxString &val)
    {
        m_ParamsFilePath = val;
    }

    /** Access m_PredictandDBFilePath
     * \return The current value of m_PredictandDBFilePath
     */
    wxString GetPredictandDBFilePath()
    {
        return m_PredictandDBFilePath;
    }

    /** Set m_PredictandDBFilePath
     * \param val New value to set
     */
    void SetPredictandDBFilePath(const wxString &val)
    {
        m_PredictandDBFilePath = val;
    }

    /** Access m_PredictandDB
     * \return The current m_PredictandDB pointer
     */
    asDataPredictand* GetPredictandDB()
    {
        return m_PredictandDB;
    }

    /** Set m_PredictandDB
     * \param pDB Pointer to the DB
     */
    void SetPredictandDB(asDataPredictand* pDB)
    {
        m_PredictandDB = pDB;
    }

    /** Access m_PredictorDataDir
     * \return The current value of m_PredictorDataDir
     */
    wxString GetPredictorDataDir()
    {
        return m_PredictorDataDir;
    }

    /** Set m_PredictorDataDir
     * \param val New value to set
     */
    void SetPredictorDataDir(const wxString &val)
    {
        m_PredictorDataDir = val;
    }

protected:
    bool m_Cancel;
    wxString m_ParamsFilePath;
    wxString m_PredictandDBFilePath;
    wxString m_PredictorDataDir;
    // TODO (Pascal#5#): Make it compatible for temperature predictand DB.
    asDataPredictand* m_PredictandDB;

private:
};

#endif // ASMETHODSTANDARD_H
