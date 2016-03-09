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
    m_originalProviderStart = 0.0;
    m_originalProviderEnd = 0.0;
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

bool asDataPredictorArchive::ClipToArea(asGeoAreaCompositeGrid *desiredArea)
{
    double Xmin = desiredArea->GetAbsoluteXmin();
    double Xmax = desiredArea->GetAbsoluteXmax();
    wxASSERT(m_axisLon.size()>0);
    int XstartIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size()-1], Xmin, 0.0, asHIDE_WARNINGS);
    int XendIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size()-1], Xmax, 0.0, asHIDE_WARNINGS);
    if (XstartIndex<0)
    {
        XstartIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size()-1], Xmin+desiredArea->GetAxisXmax());
        XendIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size()-1], Xmax+desiredArea->GetAxisXmax());
        if (XstartIndex<0 || XendIndex<0)
        {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                        Xmin+desiredArea->GetAxisXmax(), Xmax+desiredArea->GetAxisXmax(), m_axisLon[0], m_axisLon[m_axisLon.size()-1] ));
            return false;
        }
    }
    if (XstartIndex<0 || XendIndex<0)
    {

        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                    Xmin, Xmax, m_axisLon[0], m_axisLon[m_axisLon.size()-1] ));
        return false;
    }
    int Xlength = XendIndex-XstartIndex+1;

    double Ymin = desiredArea->GetAbsoluteYmin();
    double Ymax = desiredArea->GetAbsoluteYmax();
    wxASSERT(m_axisLat.size()>0);
    int YstartIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size()-1], Ymin, 0.0, asHIDE_WARNINGS);
    int YendIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size()-1], Ymax, 0.0, asHIDE_WARNINGS);
    if (XstartIndex<0)
    {
        YstartIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size()-1], Ymin+desiredArea->GetAxisYmax());
        YendIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size()-1], Ymax+desiredArea->GetAxisYmax());
        if (YstartIndex<0 || YendIndex<0)
        {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                        Ymin+desiredArea->GetAxisYmax(), Ymax+desiredArea->GetAxisYmax(), m_axisLat[0], m_axisLat[m_axisLat.size()-1] ));
            return false;
        }
    }
    if (YstartIndex<0 || YendIndex<0)
    {
        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                    Ymin, Ymax, m_axisLat[0], m_axisLat[m_axisLat.size()-1] ));
        return false;
    }

    int YstartIndexReal = wxMin(YstartIndex, YendIndex);
    int Ylength = std::abs(YendIndex-YstartIndex)+1;

    // Check if already the correct size
    if (YstartIndexReal==0 && XstartIndex==0 && Ylength==m_axisLat.size() && Xlength==m_axisLon.size() )
    {
        if (IsPreprocessed())
        {
            if(m_data[0].cols()==m_axisLon.size() && m_data[0].rows()==2*m_axisLat.size() )
            {
                // Nothing to do
                return true;
            }
            else
            {
                // Clear axes
                Array1DFloat newAxisLon(Xlength);
                for (int i=0; i<Xlength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Ylength);
                for (int i=0; i<2*Ylength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();
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
            if (method.IsSameAs("Gradients"))
            {
                VArray2DFloat originalData = m_data;

                if(originalData[0].cols()!=m_axisLon.size() || originalData[0].rows()!=2*m_axisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d", (int)originalData[0].cols(), (int)m_axisLon.size(), (int)originalData[0].rows(), (int)m_axisLat.size()));
                    return false;
                }

                /*
                Illustration of the data arrangement
                    x = data
                    o = 0

                    xxxxxxxxxxx
                    xxxxxxxxxxx
                    xxxxxxxxxxx
                    ooooooooooo____
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                */

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    Array2DFloat dat1 = originalData[i].block(YstartIndexReal,XstartIndex,Ylength-1,Xlength);
                    Array2DFloat dat2 = originalData[i].block(YstartIndexReal+m_axisLat.size(),XstartIndex,Ylength,Xlength-1);
                    Array2DFloat datMerged = Array2DFloat::Zero(2*Ylength, Xlength); // Needs to be 0-filled for further simplification.
                    datMerged.block(0,0,Ylength-1,Xlength) = dat1;
                    datMerged.block(Ylength,0,Ylength,Xlength-1) = dat2;
                    m_data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Xlength);
                for (int i=0; i<Xlength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Ylength);
                for (int i=0; i<2*Ylength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            }
            else if (method.IsSameAs("FormerHumidityIndex"))
            {
                VArray2DFloat originalData = m_data;

                if(originalData[0].cols()!=m_axisLon.size() || originalData[0].rows()!=2*m_axisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d", (int)originalData[0].cols(), (int)m_axisLon.size(), (int)originalData[0].rows(), (int)m_axisLat.size()));
                    return false;
                }

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    Array2DFloat dat1 = originalData[i].block(YstartIndexReal,XstartIndex,Ylength,Xlength);
                    Array2DFloat dat2 = originalData[i].block(YstartIndexReal+m_axisLat.size(),XstartIndex,Ylength,Xlength);
                    Array2DFloat datMerged(2*Ylength, Xlength);
                    datMerged.block(0,0,Ylength,Xlength) = dat1;
                    datMerged.block(Ylength,0,Ylength,Xlength) = dat2;
                    m_data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Xlength);
                for (int i=0; i<Xlength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Ylength);
                for (int i=0; i<2*Ylength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            }
            else if (method.IsSameAs("Multiply") || method.IsSameAs("Multiplication") || method.IsSameAs("HumidityFlux"))
            {
                VArray2DFloat originalData = m_data;

                if(originalData[0].cols()!=m_axisLon.size() || originalData[0].rows()!=m_axisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d", (int)originalData[0].cols(), (int)m_axisLon.size(), (int)originalData[0].rows(), (int)m_axisLat.size()));
                    return false;
                }

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    m_data[i] = originalData[i].block(YstartIndexReal,XstartIndex,Ylength,Xlength);
                }

                Array1DFloat newAxisLon(Xlength);
                for (int i=0; i<Xlength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Ylength);
                for (int i=0; i<2*Ylength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            }
            else
            {
                asLogError(_("Wrong proprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    VArray2DFloat originalData = m_data;
    for (unsigned int i=0; i<originalData.size(); i++)
    {
        m_data[i] = originalData[i].block(YstartIndexReal,XstartIndex,Ylength,Xlength);
    }

    Array1DFloat newAxisLon(Xlength);
    for (int i=0; i<Xlength; i++)
    {
        newAxisLon[i] = m_axisLon[XstartIndex+i];
    }
    m_axisLon = newAxisLon;

    Array1DFloat newAxisLat(Ylength);
    for (int i=0; i<Ylength; i++)
    {
        newAxisLat[i] = m_axisLat[YstartIndexReal+i];
    }
    m_axisLat = newAxisLat;

    m_latPtsnb = m_axisLat.size();
    m_lonPtsnb = m_axisLon.size();

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
    if (timeArray.GetFirst()<m_originalProviderStart)
    {
        asLogError(wxString::Format(_("The requested date (%s) is anterior to the beginning of the original dataset (%s)."),
                                    asTime::GetStringTime(timeArray.GetFirst(), YYYYMMDD),
                                    asTime::GetStringTime(m_originalProviderStart, YYYYMMDD)));
        return false;
    }
    if (!asTools::IsNaN(m_originalProviderEnd))
    {
        if (timeArray.GetLast()>m_originalProviderEnd)
        {
            asLogError(wxString::Format(_("The requested date (%s) is posterior to the end of the original dataset (%s)."),
                                        asTime::GetStringTime(timeArray.GetLast(), YYYYMMDD),
                                        asTime::GetStringTime(m_originalProviderEnd, YYYYMMDD)));
            return false;
        }
    }

    // Check the time steps
    if ((timeArray.GetTimeStepDays()>0) && (m_timeStepHours/24.0>timeArray.GetTimeStepDays()))
    {
        asLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }
    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays()/(m_timeStepHours/24.0), &intpart);
    if (fractpart>0.0000001)
    {
        asLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }
    fractpart = modf((timeArray.GetFirstDayHour()-m_firstTimeStepHours)/m_timeStepHours, &intpart);
    if (fractpart>0.0000001)
    {
        asLogError(wxString::Format(_("The desired start (%gh) is not coherent with the data properties."),
                                    timeArray.GetFirstDayHour()));
        return false;
    }

    return true;
}
