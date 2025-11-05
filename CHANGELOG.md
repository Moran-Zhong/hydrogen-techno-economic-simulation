# Changelog

All notable changes to this enhanced version of HOPP will be documented in this file.

## [Enhanced Edition] - 2025-01-09

### Added
- **Multi-Wind Farm Support**: New `MultiWindPlant` class for simulating multiple wind farms
  - Support for different wind resources per farm
  - Individual farm configuration and management
  - Aggregated power generation profiles
  
- **Advanced Optimization Algorithms**: Extended `SystemOptimizer` class
  - Nelder-Mead simplex optimization
  - Differential Evolution (DE) global optimization
  - Genetic Algorithm (GA) optimization
  - Parallel processing capabilities with LRU caching
  
- **Comprehensive Test Suites**: Real-world optimization scenarios
  - `test.py`: Multi-wind farm ratio optimization
  - `test2.py`: Dual-location wind farm modeling
  - Parallel optimization workflows
  
### Changed
- **Full Internationalization**: All Chinese comments and documentation translated to English
  - Code comments and docstrings
  - Debug output and error messages
  - User interface text
  
- **Enhanced Documentation**: Improved README with usage examples
- **Code Quality**: Cleaned up deprecated code blocks and improved error handling

### Technical Details
- Compatible with Python 3.10 and 3.11
- Maintains full backward compatibility with original HOPP
- Added support for Excel output formatting
- Improved logging and debugging capabilities

### Files Modified
- `hopp/simulation/technologies/wind/multi_wind_plant.py`
- `hopp/tools/optimization/system_optimizer.py`
- `test.py`, `test2.py` - Enhanced test scripts
- Various documentation and configuration files

---

Based on NREL's HOPP v3.2.0 with additional optimization and multi-wind farm capabilities.