#include "asThreadPreloadData.h"

asThreadPreloadData::asThreadPreloadData(asMethodCalibrator* optimizer, asParametersScoring &params, int i_step, int i_ptor, int i_dat)
:
asThread()
{
    m_status = Initializing;

    m_type = asThread::PreloadData;

    m_optimizer = optimizer; // copy pointer
    m_params = params;
    m_iStep = i_step;
    m_iProt = i_ptor;
    m_iDat = i_dat;

    m_status = Waiting;
}

asThreadPreloadData::~asThreadPreloadData()
{
    //dtor
}

wxThread::ExitCode asThreadPreloadData::Entry()
{
    m_status = Working;

    if (!m_params.NeedsPreprocessing(m_iStep, m_iProt)) {
        if (!m_optimizer->PreloadDataWithoutPreprocessing(m_params, m_iStep, m_iProt, m_iDat)) {
            return NULL;
        }
    } else {
        if (!m_optimizer->PreloadDataWithPreprocessing(m_params, m_iStep, m_iProt)) {
            return NULL;
        }
    }

    m_status = Done;

    return 0;
}
