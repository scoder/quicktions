PYTHON?=python
PKG_ROOT?=quicktions
VERSION?=$(shell sed -ne "s|^__version__\s*=\s*'\([^']*\)'.*|\1|p" $(PKG_ROOT)/quicktions.pyx)
PACKAGE=quicktions
WITH_CYTHON := $(shell python -c 'from Cython.Build import cythonize' 2>/dev/null && echo "--with-cython")

MANYLINUX_IMAGE_X86_64=quay.io/pypa/manylinux1_x86_64
MANYLINUX_IMAGE_686=quay.io/pypa/manylinux1_i686

.PHONY: all local sdist test clean realclean

all:  local

local:
	${PYTHON} setup.py build_ext --inplace $(WITH_CYTHON)

sdist: dist/$(PACKAGE)-$(VERSION).tar.gz

dist/$(PACKAGE)-$(VERSION).tar.gz:
	$(PYTHON) setup.py sdist $(WITH_CYTHON)

test: local
	PYTHONPATH=$(PKG_ROOT) $(PYTHON) $(PKG_ROOT)/test_fractions.py

clean:
	rm -fr build $(PKG_ROOT)/*.so

realclean: clean
	rm -fr $(PKG_ROOT)/*.c $(PKG_ROOT)/*.html

wheel_manylinux: wheel_manylinux64 wheel_manylinux32

wheel_manylinux32 wheel_manylinux64: dist/$(PACKAGE)-$(VERSION).tar.gz
	echo "Building wheels for $(PACKAGE) $(VERSION)"
	mkdir -p wheelhouse_$(subst wheel_,,$@)
	time docker run --rm -t \
		-v $(shell pwd):/io \
		-e CFLAGS="-O3 -g0 -mtune=generic -pipe -fPIC" \
		-e LDFLAGS="$(LDFLAGS) -fPIC" \
		-e WHEELHOUSE=wheelhouse_$(subst wheel_,,$@) \
		$(if $(patsubst %32,,$@),$(MANYLINUX_IMAGE_X86_64),$(MANYLINUX_IMAGE_686)) \
		bash -c 'for PYBIN in /opt/python/*/bin; do \
		    $$PYBIN/python -V; \
		    { $$PYBIN/pip wheel -w /io/$$WHEELHOUSE /io/$< & } ; \
		    done; wait; \
		    for whl in /io/$$WHEELHOUSE/$(PACKAGE)-$(VERSION)-*-linux_*.whl; do auditwheel repair $$whl -w /io/$$WHEELHOUSE; done'
