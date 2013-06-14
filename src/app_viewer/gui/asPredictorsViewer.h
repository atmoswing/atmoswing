#ifndef ASPREDICTORSVIEWER_H
#define ASPREDICTORSVIEWER_H

#include "vroomgis.h"
#include "asIncludes.h"

class asPredictorsManager;

class asPredictorsViewer
{
public:
    //!< The data type
    enum DisplayType
    {
        Pressure,
        Wind,
        RelativeHumidity
    };
    enum ListEntries
    {
        Hgt1000,
        Hgt500,
        Wnd1000,
        Wnd500,
        Rhum850
    };
    asPredictorsViewer(wxWindow* parent, vrLayerManager *layerManager, asPredictorsManager *predictorsManager, vrViewerLayerManager *viewerLayerManagerTarget, vrViewerLayerManager *viewerLayerManagerAnalog, wxCheckListBox *checkListPredictors);
    virtual ~asPredictorsViewer();
    void Redraw(double targetDate, double analogDate);

protected:

private:
    wxWindow* m_Parent;
    vrLayerManager *m_LayerManager;
    asPredictorsManager *m_PredictorsManager;
	vrViewerLayerManager *m_ViewerLayerManagerTarget;
	vrViewerLayerManager *m_ViewerLayerManagerAnalog;
	wxCheckListBox *m_CheckListPredictors;
	wxArrayString m_DataListString;
	VectorInt m_DataListType;
	VectorInt m_DataListEntries;
	double m_TargetDate;
	double m_AnalogDate;
};

#endif // ASPREDICTORSVIEWER_H
