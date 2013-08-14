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
    asDataPredictor();

    /** Default destructor */
    virtual ~asDataPredictor();

    /** Method to load a tensor of data for a given area and a given time array
     * \param area The desired area
     * \param timeArray The desired time array
     */
    virtual bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    /** Set m_Data: data[time](lat,lon)
     * \return The new value of m_Data
     */
    bool SetData(VArray2DFloat &val);

    /** Access m_Data: data[time](lat,lon)
     * \return The current value of m_Data
     */
    VArray2DFloat& GetData()
    {
        wxASSERT(m_Data.size()>0);
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


    /** Access m_SizeTime
     * \return The current value of m_SizeTime
     */
    int GetSizeTime()
    {
        return m_SizeTime;
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
        return m_Time[m_SizeTime-1];
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


protected:
    float m_Level;
    Array1DDouble m_Time; //!< Member variable "m_Time"
    VArray2DFloat m_Data; //!< Member variable "m_Data"
    int m_SizeTime; //!< Member variable "m_SizeTime"
    int m_LatPtsnb; //!< Member variable "m_SizeLat"
    int m_LonPtsnb; //!< Member variable "m_SizeLon"
    Array1DFloat m_AxisLat;
    Array1DFloat m_AxisLon;
    bool m_IsPreprocessed; //!< Member variable "m_IsPreprocessed"
    bool m_CanBeClipped;
    wxString m_PreprocessMethod;

    /** Method to extract the sizes of the area and the time array
     * \param area The area
     * \param timeArray The time array
     * \return True if succeeded
     */
    bool GetSizes(asGeoAreaCompositeGrid &area, asTimeArray &timeArray);

    /** Method to resize (or reserve) the final data and time array containers
     * \return True if succeeded
     */
    bool InitContainers();

    /** Method to check the time array length
     * \param counter The counter to compare
     * \return True if correct
     */
    bool CheckTimeLength(int counter);

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

private:

};

#endif // ASDATAPREDICTOR_H
