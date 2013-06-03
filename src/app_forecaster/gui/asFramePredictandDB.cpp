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
 
#include "asFramePredictandDB.h"

#include "asDataPredictandPrecipitation.h"
#include "asDataPredictandTemperature.h"
#include "asDataPredictandLightnings.h"

asFramePredictandDB::asFramePredictandDB( wxWindow* parent, wxWindowID id )
:
asFramePredictandDBVirtual( parent, id )
{
    // Set the defaults
    wxConfigBase *pConfig = wxFileConfig::Get();
    long PredictandSelection = pConfig->Read("/PredictandDBToolbox/PredictandSelection", 0l);
    m_ChoiceData->SetSelection((int)PredictandSelection);
    wxString ReturnPeriodNorm = pConfig->Read("/PredictandDBToolbox/ReturnPeriodNorm", "10");
    m_TextCtrlReturnPeriod->SetValue(ReturnPeriodNorm);
    bool NormalizeByReturnPeriod = true;
    pConfig->Read("/PredictandDBToolbox/NormalizeByReturnPeriod", &NormalizeByReturnPeriod);
    m_CheckBoxReturnPeriod->SetValue(NormalizeByReturnPeriod);
    bool ProcessSquareRoot = false;
    pConfig->Read("/PredictandDBToolbox/ProcessSquareRoot", &ProcessSquareRoot);
    m_CheckBoxSqrt->SetValue(ProcessSquareRoot);
    wxString PredictandDataDir = pConfig->Read("/PredictandDBToolbox/PredictandDataDir", wxEmptyString);
    m_DirPickerDataDir->SetPath(PredictandDataDir);
    wxString DestinationDir = pConfig->Read("/PredictandDBToolbox/DestinationDir", wxEmptyString);
    m_DirPickerDestinationDir->SetPath(DestinationDir);
    wxString PatternsDir = pConfig->Read("/PredictandDBToolbox/PatternsDir", wxEmptyString);
    m_DirPickerPatternsDir->SetPath(PatternsDir);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePredictandDB::OnSaveDefault( wxCommandEvent& event )
{
    // Save as defaults
    wxConfigBase *pConfig = wxFileConfig::Get();
    long PredictandSelection = (long)m_ChoiceData->GetSelection();
    pConfig->Write("/PredictandDBToolbox/PredictandSelection", PredictandSelection);
    wxString ReturnPeriodNorm = m_TextCtrlReturnPeriod->GetValue();
    pConfig->Write("/PredictandDBToolbox/ReturnPeriodNorm", ReturnPeriodNorm);
    bool NormalizeByReturnPeriod = m_CheckBoxReturnPeriod->GetValue();
    pConfig->Write("/PredictandDBToolbox/NormalizeByReturnPeriod", NormalizeByReturnPeriod);
    bool ProcessSquareRoot = m_CheckBoxSqrt->GetValue();
    pConfig->Write("/PredictandDBToolbox/ProcessSquareRoot", ProcessSquareRoot);
    wxString PredictandDataDir = m_DirPickerDataDir->GetPath();
    pConfig->Write("/PredictandDBToolbox/PredictandDataDir", PredictandDataDir);
    wxString DestinationDir = m_DirPickerDestinationDir->GetPath();
    pConfig->Write("/PredictandDBToolbox/DestinationDir", DestinationDir);
    wxString PatternsDir = m_DirPickerPatternsDir->GetPath();
    pConfig->Write("/PredictandDBToolbox/PatternsDir", PatternsDir);

    pConfig->Flush();
}

void asFramePredictandDB::CloseFrame( wxCommandEvent& event )
{
    Close();
}

void asFramePredictandDB::OnDataSelection( wxCommandEvent& event )
{
    switch (m_ChoiceData->GetSelection())
    {
        case wxNOT_FOUND:
        {
            break;
        }
        case 0: // precipitation
        {
            m_PanelDataProcessing->Enable();
            break;
        }
        case 1: // temperature
        {
            //m_PanelDataProcessing->Disable();
            break;
        }
    }
}

void asFramePredictandDB::BuildDatabase( wxCommandEvent& event )
{

    asLogError("The predictandDB builder is currently not working !!");
    return;



/*
    try
    {
        // Get data processing options
        // Return period
        double valReturnPeriod = 0;
        if (m_CheckBoxReturnPeriod->GetValue())
        {
            wxString valReturnPeriodString = m_TextCtrlReturnPeriod->GetValue();
            valReturnPeriodString.ToDouble(&valReturnPeriod);
            if ( (valReturnPeriod<1) | (valReturnPeriod>1000) )
            {
                asLogError(_("The given return period is not consistent."));
                return;
            }
        }

        // Sqrt option
        int makeSqrt;
        if (m_CheckBoxSqrt->GetValue())
        {
            makeSqrt = asMAKE_SQRT;
        }
        else
        {
            makeSqrt = asDONT_MAKE_SQRT;
        }

        // Get paths
        wxString pathDataDir = m_DirPickerDataDir->GetPath();
        if (pathDataDir.IsEmpty())
        {
            asLogError(_("The given path for the data directory is empty."));
            return;
        }
        wxString pathDestinationDir = m_DirPickerDestinationDir->GetPath();
        if (pathDestinationDir.IsEmpty())
        {
            asLogError(_("The given path for the output destination is empty."));
            return;
        }
        wxString pathPatternsDir = m_DirPickerPatternsDir->GetPath();
        if (pathPatternsDir.IsEmpty())
        {
            asLogError(_("The given path for the patterns directory is empty."));
            return;
        }

        switch (m_ChoiceData->GetSelection())
        {
            case wxNOT_FOUND:
            {
                asLogError(_("Wrong option selection."));
                break;
            }
            case 0: // Stations daily precipitation
            {
                // Instiantiate a predictand object
                asDataPredictandPrecipitation predictand(StationsDailyPrecipitation);
                predictand.BuildPredictandDB((float) valReturnPeriod, makeSqrt, wxEmptyString, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 1: // Stations 6-hourly series of daily precipitation
            {
                asDataPredictandPrecipitation predictand(Stations6HourlyOfDailyPrecipitation);
                predictand.BuildPredictandDB((float) valReturnPeriod, makeSqrt, wxEmptyString, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 2: // Stations 6-hourly precipitation
            {
                asDataPredictandPrecipitation predictand(Stations6HourlyPrecipitation);
                predictand.BuildPredictandDB((float) valReturnPeriod, makeSqrt, wxEmptyString, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 3: // Regional daily precipitation
            {
                asDataPredictandPrecipitation predictand(RegionalDailyPrecipitation);
                predictand.BuildPredictandDB((float) valReturnPeriod, makeSqrt, wxEmptyString, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 4: // Research daily precipitation
            {
                asDataPredictandPrecipitation predictand(ResearchDailyPrecipitation);
                predictand.BuildPredictandDB((float) valReturnPeriod, makeSqrt, wxEmptyString, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 5: // Daily lightnings
            {
                asDataPredictandLightnings predictand(StationsDailyLightnings);
                predictand.BuildPredictandDB((float) valReturnPeriod, makeSqrt, wxEmptyString, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            default:
                asLogError(_("Data selection not defined yet."));
        }
    }
    catch(asException& e)
    {
        wxString fullMessage = e.GetFullMessage();
		if (!fullMessage.IsEmpty())
		{
			asLogError(fullMessage);
		}
    }*/
}
