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
    
    /** Set m_data: data[time](lat,lon)
     * \return The new value of m_data
     */
    bool SetData(VArray2DFloat &val);

    /** Access m_data: data[time](lat,lon)
     * \return The current value of m_data
     */
    VArray2DFloat& GetData()
    {
        wxASSERT((int)m_data.size()==(int)m_time.size());
        wxASSERT(m_data.size()>1);
        wxASSERT(m_data[0].cols()>0);
        wxASSERT(m_data[0].rows()>0);
        wxASSERT(m_data[1].cols()>0);
        wxASSERT(m_data[1].rows()>0);

        return m_data;
    }

    /** Access m_time
     * \return The current value of m_time
     */
    Array1DDouble& GetTime()
    {
        return m_time;
    }

    Array1DFloat GetAxisLon()
    {
        return m_axisLon;
    }

    Array1DFloat GetAxisLat()
    {
        return m_axisLat;
    }

    void SetDirectoryPath(wxString directoryPath)
    {
        if ( (directoryPath.Last()!='/') && (directoryPath.Last()!='\\') )
        {
            directoryPath.Append('/');
        }

        m_directoryPath = directoryPath;
    }

    wxString GetDirectoryPath()
    {
        return m_directoryPath;
    }

    /** Access the size of the time array
     * \return The current value of m_sizeTime
     */
    int GetTimeSize()
    {
        return (int)m_time.size();
    }

    /** Access m_sizeLat
     * \return The current value of m_sizeLat
     */
    int GetLatPtsnb()
    {
        return m_latPtsnb;
    }

    /** Access m_sizeLon
     * \return The current value of m_sizeLon
     */
    int GetLonPtsnb()
    {
        return m_lonPtsnb;
    }

    /** Access the first value of the time array
     * \return The first value of the time array
     */
    double GetTimeStart()
    {
        return m_time[0];
    }

    /** Access the last value of the time array
     * \return The last value of the time array
     */
    double GetTimeEnd()
    {
        return m_time[m_time.size()-1];
    }

    /** Access m_isPreprocessed
     * \return The current value of m_isPreprocessed
     */
    bool IsPreprocessed()
    {
        return m_isPreprocessed;
    }

    /** Access m_isPreprocessed
     * \return The current value of m_isPreprocessed
     */
    bool GetIsPreprocessed()
    {
        return m_isPreprocessed;
    }

    /** Set m_isPreprocessed
     * \param The new value of m_isPreprocessed
     */
    void SetIsPreprocessed(bool val)
    {
        m_isPreprocessed = val;
    }

    /** Access m_canBeClipped
     * \return The current value of m_canBeClipped
     */
    bool CanBeClipped()
    {
        return m_canBeClipped;
    }

    /** Set m_canBeClipped
     * \param The new value of m_canBeClipped
     */
    void SetCanBeClipped(bool val)
    {
        m_canBeClipped = val;
    }

    /** Access m_preprocessMethod
     * \return The current value of m_preprocessMethod
     */
    wxString GetPreprocessMethod()
    {
        return m_preprocessMethod;
    }

    /** Set m_preprocessMethod
     * \param The new value of m_preprocessMethod
     */
    void SetPreprocessMethod(wxString val)
    {
        m_preprocessMethod = val;
    }

    /** Access m_finalProviderWebsite
     * \return The current value of m_finalProviderWebsite
     */
    wxString GetFinalProviderWebsite()
    {
        return m_finalProviderWebsite;
    }

    /** Access m_finalProviderFTP
     * \return The current value of m_finalProviderFTP
     */
    wxString GetFinalProviderFTP()
    {
        return m_finalProviderFTP;
    }

    /** Access m_dataId
     * \return The current value of m_dataId
     */
    wxString GetDataId()
    {
        return m_dataId;
    }

    /** Access m_datasetName
     * \return The current value of m_datasetName
     */
    wxString GetDatasetName()
    {
        return m_datasetName;
    }

    /** Access m_xAxisStep
     * \return The current value of m_xAxisStep
     */
    double GetXaxisStep()
    {
        return m_xAxisStep;
    }

    void SetXaxisStep(const double val)
    {
        m_xAxisStep = val;
    }

    /** Access m_xAxisShift
     * \return The current value of m_xAxisShift
     */
    double GetXaxisShift()
    {
        return m_xAxisShift;
    }

    /** Access m_yAxisStep
     * \return The current value of m_yAxisStep
     */
    double GetYaxisStep()
    {
        return m_yAxisStep;
    }

    void SetYaxisStep(const double val)
    {
        m_yAxisStep = val;
    }

    /** Access m_yAxisShift
     * \return The current value of m_yAxisShift
     */
    double GetYaxisShift()
    {
        return m_yAxisShift;
    }

protected:
    wxString m_directoryPath;
    bool m_initialized;
    bool m_axesChecked;
    wxString m_dataId;
    wxString m_datasetId;
    wxString m_originalProvider;
    wxString m_finalProvider;
    wxString m_finalProviderWebsite;
    wxString m_finalProviderFTP;
    wxString m_datasetName;
    double m_timeZoneHours;
    double m_timeStepHours;
    double m_firstTimeStepHours;
    VectorDouble m_nanValues;
    DataParameter m_dataParameter;
    wxString m_fileVariableName;
    DataUnit m_unit;
    float m_xAxisStep;
    float m_yAxisStep;
    float m_xAxisShift;
    float m_yAxisShift;
    float m_level;
    Array1DDouble m_time;
    VArray2DFloat m_data;
    int m_latPtsnb; 
    int m_lonPtsnb; 
    size_t m_latIndexStep; 
    size_t m_lonIndexStep;
    size_t m_timeIndexStep;
    Array1DFloat m_axisLat;
    Array1DFloat m_axisLon;
    bool m_isPreprocessed;
    bool m_canBeClipped;
    wxString m_preprocessMethod;
    wxString m_fileAxisLatName;
    wxString m_fileAxisLonName;
    wxString m_fileAxisTimeName;
    wxString m_fileAxisLevelName;
    wxString m_fileExtension;
        
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
