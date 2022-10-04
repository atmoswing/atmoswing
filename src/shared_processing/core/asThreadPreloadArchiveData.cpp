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

#include "asThreadPreloadArchiveData.h"

#include "asParameters.h"

asThreadPreloadArchiveData::asThreadPreloadArchiveData(asMethodStandard* method, asParameters* params, int iStep,
                                                       int iPtor, int i)
    : asThread(asThread::PreloadData),
      m_method(method),  // copy pointer
      m_params(params),
      m_iStep(iStep),
      m_iProt(iPtor),
      m_iDat(i) {}

asThreadPreloadArchiveData::~asThreadPreloadArchiveData() {}

wxThread::ExitCode asThreadPreloadArchiveData::Entry() {
    if (!m_params->NeedsPreprocessing(m_iStep, m_iProt)) {
        if (!m_method->PreloadArchiveDataWithoutPreprocessing(m_params, m_iStep, m_iProt, m_iDat)) {
            return (wxThread::ExitCode)-1;
        }
    } else {
        if (!m_method->PreloadArchiveDataWithPreprocessing(m_params, m_iStep, m_iProt)) {
            return (wxThread::ExitCode)-1;
        }
    }

    return (wxThread::ExitCode)0;
}
