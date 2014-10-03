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
 
#ifndef ASDATAPREDICTOR_H
#define ASDATAPREDICTOR_H

#include <asIncludes.h>

class asTimeArray;
class asGeo;
class asGeoAreaCompositeGrid;


class asDataPredictor: public wxObject
{
public:

    /** Default constructor */
    asDataPredictor(const wxString &dataId);

    /** Default destructor */
    virtual ~asDataPredictor();
    
    virtual bool Init() = 0;

    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray);
    bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray);
    bool Load(asGeoAreaCompositeGrid &desiredArea, double date);
    bool Load(asGeoAreaCompositeGrid *desiredArea, double date);
    bool LoadFullArea(double date, float level);

    bool Inline();
    
    /** Set m_Data: data[time](lat,lon)
     * \return The new value of m_Data
     */
    bool SetData(VArray2DFloat &val);

    /** Access m_Data: data[time](lat,lon)
     * \return The current value of m_Data
     */
    VArray2DFloat& GetData()
    {
        wxASSERT((int)m_Data.size()==(int)m_Time.size());
        wxASSERT(m_Data.size()>1);
        wxASSERT(m_Data[0].cols()>0);
        wxASSERT(m_Data[0].rows()>0);
        wxASSERT(m_Data[1].cols()>0);
        wxASSERT(m_Data[1].rows()>0);

        return m_Data;
    }

    /** Access m_Time
     * \return The current value of m_Time
     */
    Array1DDouble& GetTime()
    {
        return m_Time;
    }

    Array1DFloat GetAxisLon()
    {
        return m_AxisLon;
    }

    Array1DFloat GetAxisLat()
    {
        return m_AxisLat;
    }

    void SetDirectoryPath(wxString directoryPath)
    {
        if ( (directoryPath.Last()!='/') && (directoryPath.Last()!='\\') )
        {
            directoryPath.Append('/');
        }

        m_DirectoryPath = directoryPath;
    }

    wxString GetDirectoryPath()
    {
        return m_DirectoryPath;
    }

    /** Access the size of the time array
     * \return The current value of m_SizeTime
     */
    int GetTimeSize()
    {
        return (int)m_Time.size();
    }

    /** Access m_SizeLat
     * \return The current value of m_SizeLat
     */
    int GetLatPtsnb()
    {
        return m_LatPtsnb;
    }

    /** Access m_SizeLon
     * \return The current value of m_SizeLon
     */
    int GetLonPtsnb()
    {
        return m_LonPtsnb;
    }

    /** Access the first value of the time array
     * \return The first value of the time array
     */
    double GetTimeStart()
    {
        return m_Time[0];
    }

    /** Access the last value of the time array
     * \return The last value of the time array
     */
    double GetTimeEnd()
    {
        return m_Time[m_Time.size()-1];
    }

    /** Access m_IsPreprocessed
     * \return The current value of m_IsPreprocessed
     */
    bool IsPreprocessed()
    {
        return m_IsPreprocessed;
    }

    /** Access m_IsPreprocessed
     * \return The current value of m_IsPreprocessed
     */
    bool GetIsPreprocessed()
    {
        return m_IsPreprocessed;
    }

    /** Set m_IsPreprocessed
     * \param The new value of m_IsPreprocessed
     */
    void SetIsPreprocessed(bool val)
    {
        m_IsPreprocessed = val;
    }

    /** Access m_CanBeClipped
     * \return The current value of m_CanBeClipped
     */
    bool CanBeClipped()
    {
        return m_CanBeClipped;
    }

    /** Set m_CanBeClipped
     * \param The new value of m_CanBeClipped
     */
    void SetCanBeClipped(bool val)
    {
        m_CanBeClipped = val;
    }

    /** Access m_PreprocessMethod
     * \return The current value of m_PreprocessMethod
     */
    wxString GetPreprocessMethod()
    {
        return m_PreprocessMethod;
    }

    /** Set m_PreprocessMethod
     * \param The new value of m_PreprocessMethod
     */
    void SetPreprocessMethod(wxString val)
    {
        m_PreprocessMethod = val;
    }

    /** Access m_FinalProviderWebsite
     * \return The current value of m_FinalProviderWebsite
     */
    wxString GetFinalProviderWebsite()
    {
        return m_FinalProviderWebsite;
    }

    /** Access m_FinalProviderFTP
     * \return The current value of m_FinalProviderFTP
     */
    wxString GetFinalProviderFTP()
    {
        return m_FinalProviderFTP;
    }

    /** Access m_DataId
     * \return The current value of m_DataId
     */
    wxString GetDataId()
    {
        return m_DataId;
    }

    /** Access m_DatasetName
     * \return The current value of m_DatasetName
     */
    wxString GetDatasetName()
    {
        return m_DatasetName;
    }

    /** Access m_UaxisStep
     * \return The current value of m_UaxisStep
     */
    double GetUaxisStep()
    {
        return m_UaxisStep;
    }

    /** Access m_UaxisShift
     * \return The current value of m_UaxisShift
     */
    double GetUaxisShift()
    {
        return m_UaxisShift;
    }

    /** Access m_VaxisStep
     * \return The current value of m_VaxisStep
     */
    double GetVaxisStep()
    {
        return m_VaxisStep;
    }

    /** Access m_VaxisShift
     * \return The current value of m_VaxisShift
     */
    double GetVaxisShift()
    {
        return m_VaxisShift;
    }

    /** Access m_CoordinateSystem
     * \return The current value of m_CoordinateSystem
     */
    CoordSys GetCoordSys()
    {
        return m_CoordinateSystem;
    }

protected:
    wxString m_DirectoryPath;
    bool m_Initialized;
    bool m_AxesChecked;
    wxString m_DataId;
    wxString m_DatasetId;
    wxString m_OriginalProvider;
    wxString m_FinalProvider;
    wxString m_FinalProviderWebsite;
    wxString m_FinalProviderFTP;
    wxString m_DatasetName;
    double m_TimeZoneHours;
    double m_TimeStepHours;
    double m_FirstTimeStepHours;
    VectorDouble m_NanValues;
    CoordSys m_CoordinateSystem;
    DataParameter m_DataParameter;
    wxString m_FileVariableName;
    DataUnit m_Unit;
    float m_UaxisStep;
    float m_VaxisStep;
    float m_UaxisShift;
    float m_VaxisShift;
    float m_Level;
    Array1DDouble m_Time;
    VArray2DFloat m_Data;
    int m_LatPtsnb; 
    int m_LonPtsnb; 
    size_t m_LatIndexStep; 
    size_t m_LonIndexStep;
    size_t m_TimeIndexStep;
    Array1DFloat m_AxisLat;
    Array1DFloat m_AxisLon;
    bool m_IsPreprocessed;
    bool m_CanBeClipped;
    wxString m_PreprocessMethod;
        
    /** Method to check the time array compatibility with the data
     * \param timeArray The time array to check
     * \return True if compatible with the data
     */
    virtual bool CheckTimeArray(asTimeArray &timeArray)
    {
        return false;
    }

    virtual bool ExtractFromFiles(asGeoAreaCompositeGrid *& dataArea, asTimeArray &timeArray, VVArray2DFloat &compositeData)
    {
        return false;
    }

    /** Method to merge data composites
     * \param compositeData The composite data to merge
     * \param area The corresponding composite area
     * \return True if succeeded
     */
    bool MergeComposites(VVArray2DFloat &compositeData, asGeoAreaCompositeGrid *area);

    /** Method to interpolate data on a different grid
     * \param dataArea The composite grids area corresponding to the data
     * \param desiredArea The desired composite grids area
     * \return True if succeeded
     */
    bool InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid* CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid* AdjustAxes(asGeoAreaCompositeGrid *dataArea, Array1DFloat &axisDataLon, Array1DFloat &axisDataLat, VVArray2DFloat &compositeData);


private:

};

#endif // ASDATAPREDICTOR_H
