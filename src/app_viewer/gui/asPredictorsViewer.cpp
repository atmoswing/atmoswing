#include "asPredictorsViewer.h"

#include "asPredictorsManager.h"
#include "vrlayerraster.h"
#include "vrLayerRasterPressure.h"
#include "vrlayervector.h"
#include "vrrender.h"


asPredictorsViewer::asPredictorsViewer(wxWindow* parent, vrLayerManager *layerManager, asPredictorsManager *predictorsManager, vrViewerLayerManager *viewerLayerManagerTarget, vrViewerLayerManager *viewerLayerManagerAnalog, wxCheckListBox *checkListPredictors)
{
    m_Parent = parent;
    m_LayerManager = layerManager;
    m_PredictorsManager = predictorsManager;
	m_ViewerLayerManagerTarget = viewerLayerManagerTarget;
	m_ViewerLayerManagerAnalog = viewerLayerManagerAnalog;
	m_CheckListPredictors = checkListPredictors;
	m_TargetDate = -1;
	m_AnalogDate = -1;

    // According to ListEntries
	m_DataListString.Add(_("Hgt 1000 hPa"));
	m_DataListString.Add(_("Hgt 500 hPa"));
	m_DataListString.Add(_("Wind 1000 hPa"));
	m_DataListString.Add(_("Wind 500 hPa"));
	m_DataListString.Add(_("Rhum 850 hPa"));
	m_CheckListPredictors->Set(m_DataListString);
	m_CheckListPredictors->Layout();

	m_DataListEntries.push_back(Hgt1000);
	m_DataListEntries.push_back(Hgt500);
	m_DataListEntries.push_back(Wnd1000);
	m_DataListEntries.push_back(Wnd500);
	m_DataListEntries.push_back(Rhum850);

	m_DataListType.push_back(Pressure);
	m_DataListType.push_back(Pressure);
	m_DataListType.push_back(Wind);
	m_DataListType.push_back(Wind);
	m_DataListType.push_back(RelativeHumidity);

}

asPredictorsViewer::~asPredictorsViewer()
{
    //dtor
}


void asPredictorsViewer::Redraw(double targetDate, double analogDate)
{
    wxString layerName;
    //int type;
/*




    type = asPredictorsViewer::Hgt1000;
    layerName = "Hgt1000";

    Array2DFloat* data;
    Array1DFloat* axisLon, *axisLat;
    if(!m_PredictorsManager->GetArchiveData(analogDate, type, data, axisLon, axisLat))
    {
        asLogError(_("Couldn't load the archive data."));
        return;
    }









    // Create a memory layer
    wxFileName memoryLayerName ("", layerName, "memory");
    wxASSERT(memoryLayerName.GetExt() == "memory");

    // Check if memory layer already added
    m_ViewerLayerManagerTarget->FreezeBegin();
    m_ViewerLayerManagerAnalog->FreezeBegin();
	for (int i = 0; i < m_ViewerLayerManagerTarget->GetCount(); i++)
    {
		if (m_ViewerLayerManagerTarget->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerName)
        {
			vrRenderer *renderer = m_ViewerLayerManagerTarget->GetRenderer(i);
			vrLayer *layer = renderer->GetLayer();
			wxASSERT(renderer);
			m_ViewerLayerManagerTarget->Remove(renderer);
			// Close layer
			m_LayerManager->Close(layer);
		}
	}
	for (int i = 0; i < m_ViewerLayerManagerAnalog->GetCount(); i++)
    {
		if (m_ViewerLayerManagerAnalog->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerName)
        {
			vrRenderer *renderer = m_ViewerLayerManagerAnalog->GetRenderer(i);
			vrLayer *layer = renderer->GetLayer();
			wxASSERT(renderer);
			m_ViewerLayerManagerAnalog->Remove(renderer);
			// Close layer
			m_LayerManager->Close(layer);
		}
	}














    // Create the layer
    vrLayerRasterPressure * layer = new vrLayerRasterPressure();
    wxASSERT(layer);
    if(layer->CreateInMemory(data, axisLon, axisLat)==false)
    {
        wxFAIL;
        m_ViewerLayerManagerTarget->FreezeEnd();
        m_ViewerLayerManagerAnalog->FreezeEnd();
        return;
    }

    wxASSERT(layer);
    m_LayerManager->Add(layer);


    // Change default render
    vrRenderRaster * render = new vrRenderRaster();

    m_ViewerLayerManagerTarget->Add(-1, layer, render);
    m_ViewerLayerManagerAnalog->Add(-1, layer, render);
    m_ViewerLayerManagerTarget->FreezeEnd();
    m_ViewerLayerManagerAnalog->FreezeEnd();

    wxLogMessage("OKOK");





*/


    /*

            // Length of the lead time
            int leadTimeSize = forecast->GetTargetDatesLength();

            // Adding size field
            OGRFieldDefn fieldLeadTimeSize ("leadtimesize", OFTReal);
            layer->AddField(fieldLeadTimeSize);

            // Adding a field for every lead time
            for (int i=0; i<leadTimeSize; i++)
            {
                OGRFieldDefn fieldLeadTime (wxString::Format("leadtime%d", i), OFTReal);
                layer->AddField(fieldLeadTime);
            }

            // Adding features to the layer
            for (int i_stat=0; i_stat<forecast->GetStationsNb(); i_stat++)
            {
                OGRPoint station;
                station.setX( forecast->GetStationLocCoordU(i_stat) );
                station.setY( forecast->GetStationLocCoordV(i_stat) );

                // Field container
                wxArrayDouble data;
                data.Add((double)leadTimeSize);

                // For normalization by the return period
                double factor = 1;
                if (returnPeriod!=0)
                {
                    float precip = forecast->GetDailyPrecipitationForReturnPeriod(i_stat, indexReturnPeriod);
                    wxASSERT(precip>0);
                    wxASSERT(precip<500);
                    factor = 1.0/precip;
                    wxASSERT(factor>0);
                }

                // Loop over the lead times
                for (int i_leadtime=0; i_leadtime<leadTimeSize; i_leadtime++)
                {
                    Array1DFloat values = forecast->GetAnalogsValuesGross(i_leadtime, i_stat);

                    if(asTools::HasNaN(&values[0], &values[values.size()-1]))
                    {
                        data.Add(NaNDouble);
                    }
                    else
                    {
                        if (percentile>=0)
                        {
                            double forecastVal = asTools::Percentile(&values[0], &values[values.size()-1], percentile);
                            wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                            forecastVal *= factor;
                            data.Add(forecastVal);
                        }
                        else
                        {
                            // Interpretatio
                            double forecastVal = 0;
                            double forecastVal30 = asTools::Percentile(&values[0], &values[values.size()-1], 0.3f);
                            double forecastVal60 = asTools::Percentile(&values[0], &values[values.size()-1], 0.6f);
                            double forecastVal90 = asTools::Percentile(&values[0], &values[values.size()-1], 0.9f);

                            if(forecastVal60==0)
                            {
                                forecastVal = 0;
                            }
                            else if(forecastVal30>0)
                            {
                                forecastVal = forecastVal90;
                            }
                            else
                            {
                                forecastVal = forecastVal60;
                            }

                            wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                            forecastVal *= factor;
                            data.Add(forecastVal);
                        }
                    }
                }

                layer->AddFeature(&station, &data);
            }

            wxASSERT(layer);
            m_LayerManager->Add(layer);

            // Change default render
            vrRenderVector * render = new vrRenderVector();
            render->SetSize(1);
            render->SetColorPen(*wxBLACK);

            m_ViewerLayerManager->Add(-1, layer, render);
            m_ViewerLayerManager->FreezeEnd();

            break;
        }
    case ForecastDots:
        {
            wxASSERT(m_LeadTimeIndex>=0);

            // Create the layer
            vrLayerVectorFcstDots * layer = new vrLayerVectorFcstDots();
            if(layer->Create(memoryLayerName, wkbPoint)==false)
            {
                wxFAIL;
                m_ViewerLayerManager->FreezeEnd();
                return;
            }

            // Set the maximum value
            if (m_ForecastDisplaySelection==0) // Only if the value option is selected, and not the ratio
            {
                layer->SetMaxValue(colorbarMaxValue);
                m_LayerMaxValue = colorbarMaxValue;
            }
            else
            {
                layer->SetMaxValue(1.0);
                m_LayerMaxValue = 1.0;
            }

            // Adding size field
            OGRFieldDefn fieldValueReal ("valueReal", OFTReal);
            layer->AddField(fieldValueReal);
            OGRFieldDefn fieldValueNorm ("valueNorm", OFTReal);
            layer->AddField(fieldValueNorm);

            // Adding features to the layer
            for (int i_stat=0; i_stat<forecast->GetStationsNb(); i_stat++)
            {
                OGRPoint station;
                station.setX( forecast->GetStationLocCoordU(i_stat) );
                station.setY( forecast->GetStationLocCoordV(i_stat) );

                // Field container
                wxArrayDouble data;

                // For normalization by the return period
                double factor = 1;
                if (returnPeriod!=0)
                {
                    float precip = forecast->GetDailyPrecipitationForReturnPeriod(i_stat, indexReturnPeriod);
                    wxASSERT(precip>0);
                    wxASSERT(precip<500);
                    factor = 1.0/precip;
                    wxASSERT(factor>0);
                }

                // Check available lead times
                if(forecast->GetTargetDatesLength()<=m_LeadTimeIndex)
                {
                    m_LeadTimeIndex = forecast->GetTargetDatesLength()-1;
                }

                Array1DFloat values = forecast->GetAnalogsValuesGross(m_LeadTimeIndex, i_stat);

                if(asTools::HasNaN(&values[0], &values[values.size()-1]))
                {
                    data.Add(NaNDouble); // 1st real value
                    data.Add(NaNDouble); // 2nd normalized
                }
                else
                {
                    if (percentile>=0)
                    {
                        double forecastVal = asTools::Percentile(&values[0], &values[values.size()-1], percentile);
                        wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                        data.Add(forecastVal); // 1st real value
                        forecastVal *= factor;
                        data.Add(forecastVal); // 2nd normalized
                    }
                    else
                    {
                        // Interpretatio
                        double forecastVal = 0;
                        double forecastVal30 = asTools::Percentile(&values[0], &values[values.size()-1], 0.3f);
                        double forecastVal60 = asTools::Percentile(&values[0], &values[values.size()-1], 0.6f);
                        double forecastVal90 = asTools::Percentile(&values[0], &values[values.size()-1], 0.9f);

                        if(forecastVal60==0)
                        {
                            forecastVal = 0;
                        }
                        else if(forecastVal30>0)
                        {
                            forecastVal = forecastVal90;
                        }
                        else
                        {
                            forecastVal = forecastVal60;
                        }

                        wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                        data.Add(forecastVal); // 1st real value
                        forecastVal *= factor;
                        data.Add(forecastVal); // 2nd normalized
                    }
                }

                layer->AddFeature(&station, &data);
            }

            wxASSERT(layer);
            m_LayerManager->Add(layer);

            // Change default render
            vrRenderVector * render = new vrRenderVector();
            render->SetSize(1);
            render->SetColorPen(*wxBLACK);

            m_ViewerLayerManager->Add(-1, layer, render);
            m_ViewerLayerManager->FreezeEnd();

            break;
        }
    }*/


}
