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

#include "asDataPredictorArchive.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>
#include <asDataPredictorArchiveNcepReanalysis1.h>
#include <asDataPredictorArchiveNcepReanalysis1Terranum.h>
#include <asDataPredictorArchiveNcepReanalysis1Lthe.h>
#include <asDataPredictorArchiveNcepReanalysis2.h>
#include <asDataPredictorArchiveNoaaOisst2.h>
#include <asDataPredictorArchiveNoaaOisst2Terranum.h>


asDataPredictorArchive::asDataPredictorArchive(const wxString &dataId)
:
asDataPredictor(dataId)
{

}

asDataPredictorArchive::~asDataPredictorArchive()
{

}

asDataPredictorArchive* asDataPredictorArchive::GetInstance(const wxString &datasetId, const wxString &dataId, const wxString &directory)
{
    asDataPredictorArchive* predictor = NULL;

    if (datasetId.IsSameAs("NCEP_Reanalysis_v1", false))
    {
        predictor = new asDataPredictorArchiveNcepReanalysis1(dataId);
    }
    else if (datasetId.IsSameAs("NCEP_Reanalysis_v1_terranum", false))
    {
        predictor = new asDataPredictorArchiveNcepReanalysis1Terranum(dataId);
    }
    else if (datasetId.IsSameAs("NCEP_Reanalysis_v1_lthe", false))
    {
        predictor = new asDataPredictorArchiveNcepReanalysis1Lthe(dataId);
    }
    else if (datasetId.IsSameAs("NCEP_Reanalysis_v2", false))
    {
        predictor = new asDataPredictorArchiveNcepReanalysis2(dataId);
    }
    else if (datasetId.IsSameAs("NOAA_OISST_v2", false))
    {
        predictor = new asDataPredictorArchiveNoaaOisst2(dataId);
    }
    else if (datasetId.IsSameAs("NOAA_OISST_v2_terranum", false))
    {
        predictor = new asDataPredictorArchiveNoaaOisst2Terranum(dataId);
    }
    else
    {
        asLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return NULL;
    }

    if (!directory.IsEmpty()) {
        predictor->SetDirectoryPath(directory);
    }

    if(!predictor->Init())
    {
        asLogError(_("The predictor did not initialize correctly."));
        return NULL;
    }

    return predictor;
}

bool asDataPredictorArchive::Init()
{
    return false;
}

bool asDataPredictorArchive::LoadFullArea(double date, float level)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();
    m_Level = level;

    return Load(NULL, timeArray);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid &desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(&desiredArea, timeArray);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid *desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(desiredArea, timeArray);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
{
    return false;
}

bool asDataPredictorArchive::ClipToArea(asGeoAreaCompositeGrid *desiredArea)
{
    double Umin = desiredArea->GetAbsoluteUmin();
    double Umax = desiredArea->GetAbsoluteUmax();
    wxASSERT(m_AxisLon.size()>0);
    int UstartIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umin, 0.0, asHIDE_WARNINGS);
    int UendIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umax, 0.0, asHIDE_WARNINGS);
    if (UstartIndex<0)
    {
        UstartIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umin+desiredArea->GetAxisUmax());
        UendIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umax+desiredArea->GetAxisUmax());
        if (UstartIndex<0 || UendIndex<0)
        {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                        Umin+desiredArea->GetAxisUmax(), Umax+desiredArea->GetAxisUmax(), m_AxisLon[0], m_AxisLon[m_AxisLon.size()-1] ));
            return false;
        }
    }
    if (UstartIndex<0 || UendIndex<0)
    {

        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                    Umin, Umax, m_AxisLon[0], m_AxisLon[m_AxisLon.size()-1] ));
        return false;
    }
    int Ulength = UendIndex-UstartIndex+1;

    double Vmin = desiredArea->GetAbsoluteVmin();
    double Vmax = desiredArea->GetAbsoluteVmax();
    wxASSERT(m_AxisLat.size()>0);
    int VstartIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmin, 0.0, asHIDE_WARNINGS);
    int VendIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmax, 0.0, asHIDE_WARNINGS);
    if (UstartIndex<0)
    {
        VstartIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmin+desiredArea->GetAxisVmax());
        VendIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmax+desiredArea->GetAxisVmax());
        if (VstartIndex<0 || VendIndex<0)
        {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                        Vmin+desiredArea->GetAxisVmax(), Vmax+desiredArea->GetAxisVmax(), m_AxisLat[0], m_AxisLat[m_AxisLat.size()-1] ));
            return false;
        }
    }
    if (VstartIndex<0 || VendIndex<0)
    {
        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                    Vmin, Vmax, m_AxisLat[0], m_AxisLat[m_AxisLat.size()-1] ));
        return false;
    }

    int VstartIndexReal = wxMin(VstartIndex, VendIndex);
    int Vlength = abs(VendIndex-VstartIndex)+1;

    // Check if already the correct size
    if (VstartIndexReal==0 && UstartIndex==0 && Vlength==m_AxisLat.size() && Ulength==m_AxisLon.size() )
    {
        if (IsPreprocessed())
        {
            if(m_Data[0].cols()==m_AxisLon.size() && m_Data[0].rows()==2*m_AxisLat.size() )
            {
                // Nothing to do
                return true;
            }
            else
            {
                // Clear axes
                Array1DFloat newAxisLon(Ulength);
                for (int i=0; i<Ulength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_AxisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Vlength);
                for (int i=0; i<2*Vlength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_AxisLat = newAxisLat;

                m_LatPtsnb = m_AxisLat.size();
                m_LonPtsnb = m_AxisLon.size();
            }
        }
        else
        {
            // Nothing to do
            return true;
        }
    }
    else
    {
        if (!CanBeClipped())
        {
            asLogError(_("The preprocessed area cannot be clipped to another area."));
            return false;
        }

        if (IsPreprocessed())
        {
            wxString method = GetPreprocessMethod();
            if (method.IsSameAs("MergeCouplesAndMultiply"))
            {
                VArray2DFloat originalData = m_Data;

                if(originalData[0].cols()!=m_AxisLon.size() || originalData[0].rows()!=2*m_AxisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_AxisLon.size() = %d, originalData[0].rows() = %d, m_AxisLat.size() = %d", (int)originalData[0].cols(), (int)m_AxisLon.size(), (int)originalData[0].rows(), (int)m_AxisLat.size()));
                    return false;
                }

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    Array2DFloat dat1 = originalData[i].block(VstartIndexReal,UstartIndex,Vlength,Ulength);
                    Array2DFloat dat2 = originalData[i].block(VstartIndexReal+m_AxisLat.size(),UstartIndex,Vlength,Ulength);
                    Array2DFloat datMerged(2*Vlength, Ulength);
                    datMerged.block(0,0,Vlength,Ulength) = dat1;
                    datMerged.block(Vlength,0,Vlength,Ulength) = dat2;
                    m_Data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Ulength);
                for (int i=0; i<Ulength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_AxisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Vlength);
                for (int i=0; i<2*Vlength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_AxisLat = newAxisLat;

                m_LatPtsnb = m_AxisLat.size();
                m_LonPtsnb = m_AxisLon.size();

                return true;

            }
            else if (method.IsSameAs("HumidityFlux"))
            {
                VArray2DFloat originalData = m_Data;

                if(originalData[0].cols()!=m_AxisLon.size() || originalData[0].rows()!=2*m_AxisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_AxisLon.size() = %d, originalData[0].rows() = %d, m_AxisLat.size() = %d", (int)originalData[0].cols(), (int)m_AxisLon.size(), (int)originalData[0].rows(), (int)m_AxisLat.size()));
                    return false;
                }

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    Array2DFloat dat1 = originalData[i].block(VstartIndexReal,UstartIndex,Vlength,Ulength);
                    Array2DFloat dat2 = originalData[i].block(VstartIndexReal+m_AxisLat.size(),UstartIndex,Vlength,Ulength);
                    Array2DFloat datMerged(2*Vlength, Ulength);
                    datMerged.block(0,0,Vlength,Ulength) = dat1;
                    datMerged.block(Vlength,0,Vlength,Ulength) = dat2;
                    m_Data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Ulength);
                for (int i=0; i<Ulength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_AxisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Vlength);
                for (int i=0; i<2*Vlength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_AxisLat = newAxisLat;

                m_LatPtsnb = m_AxisLat.size();
                m_LonPtsnb = m_AxisLon.size();

                return true;
            }
            else
            {
                asLogError(_("Wrong proprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    VArray2DFloat originalData = m_Data;
    for (unsigned int i=0; i<originalData.size(); i++)
    {
        m_Data[i] = originalData[i].block(VstartIndexReal,UstartIndex,Vlength,Ulength);
    }

    Array1DFloat newAxisLon(Ulength);
    for (int i=0; i<Ulength; i++)
    {
        newAxisLon[i] = m_AxisLon[UstartIndex+i];
    }
    m_AxisLon = newAxisLon;

    Array1DFloat newAxisLat(Vlength);
    for (int i=0; i<Vlength; i++)
    {
        newAxisLat[i] = m_AxisLat[VstartIndexReal+i];
    }
    m_AxisLat = newAxisLat;

    m_LatPtsnb = m_AxisLat.size();
    m_LonPtsnb = m_AxisLon.size();

    return true;
}

bool asDataPredictorArchive::CheckTimeArray(asTimeArray &timeArray)
{
    if (!timeArray.IsSimpleMode())
    {
        asLogError(_("The data loading only accepts time arrays in simple mode."));
        return false;
    }

    // Check against original dataset
    if (timeArray.GetFirst()<m_OriginalProviderStart)
    {
        asLogError(wxString::Format(_("The requested date (%s) is anterior to the beginning of the original dataset (%s)."),
                                    asTime::GetStringTime(timeArray.GetFirst(), YYYYMMDD).c_str(),
                                    asTime::GetStringTime(m_OriginalProviderStart, YYYYMMDD).c_str()));
        return false;
    }
    if (!asTools::IsNaN(m_OriginalProviderEnd))
    {
        if (timeArray.GetLast()>m_OriginalProviderEnd)
        {
            asLogError(wxString::Format(_("The requested date (%s) is posterior to the end of the original dataset (%s)."),
                                        asTime::GetStringTime(timeArray.GetLast(), YYYYMMDD).c_str(),
                                        asTime::GetStringTime(m_OriginalProviderEnd, YYYYMMDD).c_str()));
            return false;
        }
    }

    // Check the time steps
    if ((timeArray.GetTimeStepDays()>0) && (m_TimeStepHours/24.0>timeArray.GetTimeStepDays()))
    {
        asLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }
    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays()/(m_TimeStepHours/24.0), &intpart);
    if (fractpart>0.0000001)
    {
        asLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }
    fractpart = modf((timeArray.GetFirstDayHour()-m_FirstTimeStepHours)/m_TimeStepHours, &intpart);
    if (fractpart>0.0000001)
    {
        asLogError(wxString::Format(_("The desired start (%gh) is not coherent with the data properties."),
                                    timeArray.GetFirstDayHour()));
        return false;
    }

    return true;
}
