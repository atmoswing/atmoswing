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
 */

#ifndef AS_THREAD_H
#define AS_THREAD_H

#include <wx/thread.h>

#include "asIncludes.h"

class asThread : public wxThread {
  public:
    enum Type {
        Undefined,
        PreloadData,
        ProcessorGetAnalogsDates,
        ProcessorGetAnalogsSubDates,
        MethodOptimizerMC,
        MethodOptimizerGAs
    };

    explicit asThread(Type type = Undefined);

    ~asThread() override = default;

    ExitCode Entry() override;

    void OnExit() override;

    asThread::Type GetType() const {
        return m_type;
    }

    int GetDevice() const {
        return m_device;
    }

    void SetDevice(int val) {
        m_device = val;
    }

  protected:
    asThread::Type m_type;
    int m_device;

  private:
};

#endif
